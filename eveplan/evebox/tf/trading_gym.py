import tensorflow as tf

from evebox.actions import 

class TradingGym:
    def __init__(self, universe):
        self.universe = universe
    
    def encode_states(self, states):
        output = {}
        mask   = {}
        
        # Encode location
        output['systems'] = tf.constant(
            [
                [
                    1 if state.system == system else 0,
                    universe.distance(state.system, system)
                ]
                for system in universe.systems
            ], dtype = tf.float32
        )
        
        cargos = [dict(state.cargo) for state in states]
        
        # Encode types
        output['types'] = tf.constant(
            [
                [
                    cargo.get(type, (0, 0))[0],
                    cargo.get(type, (0, 0))[1]
                ]
                for cargo in cargos
            ]
        )
        
        # Encode orders
        maxlen = max([len(s.orders) for s in states])
        mask_orders = any([len(s.orders) != maxlen for s in states])
        
        # TODO: Check if this should really be encoded in the state, or derived from
        # global orders.
        alldata, allmasks = zip(*[encode_orders(s.orders, with_mask = True) for s in states])
        
        output['orders'] = tf.stack(
            alldata
            axis = 0
        )
        
        if mask_orders:
            