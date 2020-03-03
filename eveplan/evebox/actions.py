# --- Action calculus ---

# Wrapper that allows an action to just modify a mutable state representation
# and performs validity checking of resulting state
class ActionImpl:
    def __init__(self, f, desc = None, repr = None):
        self.wrapped = f
        self.desc = desc
        self.repr = desc if repr is None else repr
    
    def __call__(self, state):
        s = MutableState(state)
        s.last_modified_item = None
        self.wrapped(s)
        s = State(s)
        
        s.validate()
        
        return s
    
    def __repr__(self):
        return self.repr
    
    def __str__(self):
        return self.desc
        
def action(**kwargs):
    def impl(f):
        return ActionImpl(f, **kwargs)
    return impl

# --- Marker broker implementation and market actions ---

def market_transaction(state, type_id, amount, order_type):
    assert order_type in ['buy', 'sell']
    
    amount = int(amount)
    
    # For the duration of this function, we need this element to exist
    if type_id not in state.cargo:
        state.cargo[type_id] = (0, 0)
    
    # Get all possible orders in appropriate order
    candidate_orders = orders[
        #(orders.location_id == state.station) &
        (orders.system_id == state.system) &
        (orders.type_id == type_id) &
        (orders.is_buy_order == (order_type == 'sell'))
    ].sort_values('price', ascending = order_type == 'buy').reset_index()
    
    # Check that we only sell as many items as we have
    if order_type == 'sell':
        amount = min(amount, state.cargo[type_id][0])    
    
    factor = 1 if order_type == 'buy' else -1
    
    total_transfer = 0
    
    for order in candidate_orders.itertuples():
        # Check if we even need anything more
        if amount == 0:
            break
        
        # Check if we can get something from this order
        remaining = order.volume_remain
        
        if order.order_id in state.updated_orders:
            remaining = state.updated_orders[order.order_id]
                
        transfer = min(amount, remaining)
        
        # Don't buy more than we can afford or store on our ship
        if order_type == 'buy':
            transfer = min(
                transfer,
                int(state.wallet / order.price)
            )
    
            vol_left, coll_left = State(state).limits_left()
            can_store = int(min(
                vol_left / types[type_id]["volume"],
                coll_left / order.price
            ))
            
            transfer = min(
                transfer,
                can_store
            )
        
        if transfer == 0:
            continue
        
        # Conduct transaction
        delta = transfer if order_type == 'buy' else -transfer
        state.wallet -= delta * order.price
        
        (cnum, cval) = state.cargo[type_id]
        state.cargo[type_id] = (
            cnum + delta,
            cval + delta * order.price
        )
        
        # Record that the order was partially filled
        state.updated_orders[order.order_id] = remaining - transfer
        
        # Continue buying if neccessary
        amount -= transfer
        
        total_transfer += transfer
    
    #if total_transfer > 0:
    #    print('Transferred {} of {}'.format(total_transfer, types[type_id]["name"]))
    
    # Remove empty cargo categories
    if state.cargo[type_id][0] == 0:
        del state.cargo[type_id]
    
def buy(type_id, amount):
    @action(desc = 'Buy {} of {}'.format(amount, types[type_id]["name"]))
    def do_buy(s):
        market_transaction(s, type_id, amount, 'buy')
        
    return do_buy

def sell(type_id, amount):
    @action(desc = 'Sell {} of {}'.format(amount, types[type_id]["name"]))
    def do_sell(s):
        market_transaction(s, type_id, amount, 'sell')
    
    return do_sell
# --- Movement actions ---

warp_cost = 1.0

@functools.lru_cache(maxsize=None)
def warp_to_station(station_id):
    station = get_station(station_id)
    
    distances = nx.shortest_path_length(subgraph, station.system_id)
    
    @action(desc = 'Move to {} in {}'.format(station.name, systems[station.system_id]["name"]))
    def do_warp_to_station(s):
        assert s.station != station_id, 'Cannot warp from to same station'
        
        if s.system == station.system_id:
            t = warp_cost
        else:
            #t = warp_time * nx.shortest_path_length(subgraph, s.system, station.system_id)
            t = warp_cost * distances[s.system]
        
        s.station = station_id
        s.system = station.system_id
        
        s.time_left -= t
    
    return do_warp_to

@functools.lru_cache(maxsize=None)
def warp_to_system(system_id):
    distances = nx.shortest_path_length(subgraph, system_id)
    
    @action(desc = 'Move to system {}'.format(systems[system_id]["name"]))
    def do_warp_to_system(s):
        t = warp_cost * system_distance(system_id, s.system)
        
        s.system = system_id
        s.station = None
        
        s.time_left -= t
    
    return do_warp_to_system