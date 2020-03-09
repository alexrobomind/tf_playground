import copy
import pandas as pd

## Structure of the state space
class StateProp:
    def __init__(self, default, convert = (lambda x: x, lambda x: x)):
        self.convert_in = convert[0]
        self.convert_out = convert[1]
        self.default = default

def to_fixed(x):
    if x is None:
        return None
    return State(x)

def to_mutable(x):
    if x is None:
        return none
    return MutableState(x)

# Dict props should be stored as (key, value) tuples
dict_prop = (lambda x: tuple((i for i in x.items())), lambda x: {k:v for (k,v) in x})

state_props = {
    # Position of the ship
    'system' : StateProp(0),
    'station' : StateProp(None),
    
    # Configuration of the ship
    'volume_limit' : StateProp(0),
    'collateral_limit' : StateProp(0),
    
    # Resource balance
    'time_left' : StateProp(0),
    'wallet' : StateProp(0),
    'cargo' : StateProp({}, dict_prop),
    
    # Market cursors
    'market_group' : StateProp(None),
    'market_item' : StateProp(None),
    'last_modified_item' : StateProp(None),
    'updated_orders' : StateProp({}, dict_prop)
}
    
# Implementation of the state space
class State:
    def __init__(self, inp):
        for name, prop in state_props.items():
            inval = getattr(inp, name, prop.default)
            inval = prop.convert_in(inval)
            object.__setattr__(self, '_prop_' + name, inval)
            
        self._encode_cache = None
    
    def __getattr__(self, name):
        if name in state_props:
            return getattr(self, '_prop_' + name)
        else:
            raise AttributeError('Unknown attribute ' + name)
    
    def __setattr__(self, name, val):
        if name in state_props:
            raise AttributeError('Attribute {} is immutable'.format(name))
        
        object.__setattr__(self, name, val)
    
    def __hash__(self):
        h = 0
        
        for name in state_props:
            h = h ^ hash(getattr(self, name))
        
        return h
    
    def __eq__(self, other):
        for name in state_props:
            if getattr(self, name) != getattr(other, name):
                return False
        
        return True
    
    def __repr__(self):
        (vleft, cleft) = self.limits_left()
        vmax = self.volume_limit
        cmax = self.collateral_limit
        v = vmax - vleft
        c = cmax - cleft
        
        #orders_by_id = orders.loc[[k for k,v in self.updated_orders]]
        #orders_by_id['vol_mod'] = [v for k,v in self.updated_orders]
        orders = pd.DataFrame.from_dict(self.updated_orders, orient = 'index', columns = ['ID', 'Remaining'])
        
        cargo_df = pd.DataFrame.from_dict(dict(self.cargo), orient = 'index', columns = ['Amount', 'Value'])
        cargo_df.index.name = 'Item ID'
        cargo_df['Item name'] = cargo_df.index.map(lambda x : types[x]["name"])
        cargo_df = cargo_df[['Item name', 'Amount', 'Value']]
        
        def cargo_entry_info(entry):
            item_id, (amount, value) = entry
            
            item_name = types[item_id]["name"]
            
            return "{:>10} x {:<20} ({:>10.2f} ISK)".format(amount, item_name, value)
        
        def indent(x, n):
            return x.replace("\n", "\n" + " " * n)
        
        cargo_string = "\n".join([cargo_entry_info(e) for e in self.cargo])
        
        try:
            self.validate()
            validated = 'Valid'
        except Exception as e:
            validated = str(e)
            
        return """
State:
    Validation: {valid}
    
    System:     {system} ({sysid})
    Station ID: {station}
    
    Resources:
        Time:       {t:>10.2f}
        Wallet:     {wallet:>10.2f}
        Collateral: {cleft:>10.2f} / {cmax:>10.2f} ({c:>10.2f} used)
        Cargo:      {vleft:>10.2f} / {vmax:>10.2f} ({v:>10.2f} used)
        
    Cargo contents:
        {cargo}
        
    Modified orders:
        {orders}
        """.format(
            valid = validated,
            system = self.system, #systems[self.system]["name"] if self.system is not None and self.system in systems else "Unknown",
            sysid = self.system,
            station = self.station,
            
            t = self.time_left,
            wallet = self.wallet,
            c = c, cmax = cmax, cleft = cleft,
            v = v, vmax = vmax, vleft = vleft,
            
            cargo = indent(cargo_string, 8),
            #cargo  = indent(cargo_df.to_string(), 8),
            #orders = indent(orders_by_id.to_string(), 8)
            orders = orders
        )
    
    def limits_left(self):
        total_vol = 0
        total_col = 0
        
        for item, (count, value) in self.cargo:
            total_vol += types[item]["volume"] * count
            total_col += value
        
        return (self.volume_limit - total_vol, self.collateral_limit - total_col)
    
    def validate(self):
        assert self.system is not None, 'No system specified'
        #assert self.system in system_ids, 'Not in a known system'
        
        (v, c) = self.limits_left()
        assert v >= 0, 'Volume exceeded'
        assert c >= 0, 'Collateral exceeded'
        
        assert self.time_left >= 0, 'Out of time'
    
    @property
    def value(self):
        return self.wallet + sum([value for t, (count, value) in self.cargo])
        

# Helper class for incremental state modification
class MutableState:
    def __init__(self, other = None):
        if other is None:
            for name, prop in state_props.items():
                setattr(self, name, copy.copy(prop.default))
        else:
            for name, prop in state_props.items():
                setattr(self, name, prop.convert_out(getattr(other, name)))