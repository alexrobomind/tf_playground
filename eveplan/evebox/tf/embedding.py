import tensorflow as tf

from evebox.util import notqdm
    
def mask_for_gather(params, indices, idx_mask = None, params_mask = None, batch_dims = 0, axis = None):
    if axis is None:
        axis = batch_dims
    
    # We need positive axes here
    if axis < 0:
        axis = tf.rank(params) + axis

    with tf.name_scope('mask_for_gather'):
        total_shape = tf.concat([
            tf.shape(params)[:axis],
            tf.shape(indices)[n_batch_dims:]
            tf.shape(params)
        ])
        
        # Output structure: Batch, Params up to axis, axis replaced with indices, params from axis on
        idx_mask_shape = tf.concat([
            tf.shape(indices)[:batch_dims], # Batch
            tf.ones([axis - batch_dims], dtype = tf.int32), # params dimensions from batch to before axis
            tf.shape(indices)[batch_dims:], # indices from batch on
            tf.ones(tf.rank(params) - axis - 1) # params dimensiosn from after axis
        ])
        
        param_mask_shape = tf.concat([
            tf.shape(params)[:axis],
            tf.ones(tf.rank(indices) - batch_dims),
            tf.shape(params)[axis + 1:]
        ])
        
        if idx_mask is not None:
            idx_mask = tf.reshape(idx_mask, idx_mask_shape)
        
        if param_mask is not None:
            param_mask = tf.shape(param_mask, param_mask_shape)
            
        if idx_mask is not None and param_mask is not None:
            return tf.math.logical_and(idx_mask, param_mask)
        elif idx_mask is not None:
            return tf.broadcast_to(idx_mask, total_shape)
        elif param_mask is not None:
            return tf.broadcast_to(param_mask, total_shape)
        else:
            return None

class Embedding:
    def __init__(self, universe):
        self.universe = universe
    

class Embedding(tf.keras.layers.Layer):
    def __init__(self, universe, landmarks = [], tqdm = notqdm):
        self.universe = universe
        self.landmarks = []
        
        self._encode_systems(tqdm)
        self._encode_types(tqdm)
        
        self.system_notes = tf.Variable(
            tf.zeros(tf.shape(self.systems), dtype = tf.float32)
        )
        self.type_notes = tf.Variable(
            tf.zeros(tf.shape(self.types), dtype = tf.float32)
        )
    
    def _encode_systems(self, tqdm):
        #self.system_ids = tf.constant(
        #    list(self.universe.systems),
        #    dtype = tf.int32
        #)
        #
        self.systems = tf.constant(
            [
                [
                    s['security_status']
                ] + [
                    self.universe.distance(k, l)
                    for l in self.landmarks
                ]
                for k, s in tqdm(self.universe.systems.items(), desc = 'Encoding systems')
            ],
            dtype = tf.int32
        )
    
    def _encode_types(self, tqdm):
        #self.type_ids = tf.constant(
        #    list(self.universe.market_types)
        #)
        #
        self.types = tf.constant(
            [
                t['volume']
            ]
            for k, t in tqdm(self.universe.market_types.items(), desc = 'Encoding types')
        )
    
    def encode_orders(self, orders):
        tidx = {
            t : idx
            for i, t in enumerate(self.types)
        }
        sidx = {
            s : idx
            for i, s in enumerate(self.systems)
        }
        
        orders_types = tf.constant(
            self.orders['type_id'].apply(lambda x : tidx(s)).as_numpy(np.int32)
        )
        orders_systems = tf.constant(
            self.orders['system_id'].apply(lambda x : sidx(s)).as_numpy(np.int32)
        )
        
        data = self.orders.copy()
        data['is_sell_order'] = ~data['is_buy_order']
        
        orders_data = tf.constant(
            self.orders[['volume_remaining', 'is_buy_order', 'is_sell_order']].as_numpy(np.float32)
        )
        
        return (orders_types, orders_systems, orders_data)

        
    def call(self, input, mask = None):
        all_system_data = tf.concat([self.systems, self.system_notes], axis = -1)
        all_type_data   = tf.concat([self.types,   self.type_notes  ], axis = -1)
        
        output = {
            'systems' : all_system_data,
            'types' : all_type_data
        }
        
        if 'orders' in input:
            (orders_types, orders_systems, orders_data) = input['orders']
        
            all_orders_data = tf.concat([
                tf.gather(all_type_data,   orders_types),
                tf.gather(all_system_data, orders_systems),
                orders_data
            ], axis = -1)
            
            output['orders'] = all_orders_data
        
        if 'cargo' in input:
            (cargo_types, cargo_data) = input['cargo']
            
            all_cargo_data = tf.concat([
                tf.gather(all_type_data, cargo_types),
                cargo_data
            ], axis = -1)
            
            output['cargo'] = all_cargo_data
        
        return output
    
    def compute_mask(self, inputs, mask = None):
        if mask is None:
            return None
        
        output = {
            'systems' : None,
            'types' : None
        }
        
        if 'orders' in input:
            (orders_types, orders_systems, orders_data) = input['orders']
            (mask_types, mask_systems, mask_data) = mask['orders']
            
            all_orders_masks = tf.concat([
                mask_for_gather(all_type_data, orders_types, idx_mask = mask_types),
                mask_for_gather(all_system_data, orders_systems, idx_mask = mask_systems),
                mask_data
            ], axis = -1)
            
            output['orders'] = all_orders_masks
        
        if 'cargo' in input:
            (cargo_types, cargo_data) = input['cargo']
            (mask_types, mask_data) = mask['cargo']
            
            all_cargo_masks = tf.concat([
                mask_for_gather(all_type_data, cargo_types, idx_mask = mask_types),
                mask_data
            ], axis = -1)
            
            output['cargo'] = all_cargo_masks
        
        return output
            