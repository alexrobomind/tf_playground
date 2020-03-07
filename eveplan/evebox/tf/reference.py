import tensorflow as tf

class ReferenceModel(tf.keras.Model):
    """
    Reference model for the trading gym. Checks the structure & shape consistency of all inputs, and returns
    outputs of the correct shape.
    """
    def __init__(self, universe, orders, state_shape):
        super().__init__()
        
        self.n_types = len(universe.types)
        self.n_systems = len(universe.systems)
        self.n_orders = len(orders)
        self.state_shape = state_shape
        
    def call(input):
        # --- Input shape check ---
        
        input, prev_state = input
        
        # RNN state shape check
        batch_shape = tf.shape(prev_state)[:-tf.size(self.state_shape)]
        tf.assert(tf.all(
            batch_shape == tf.concat([batch_shape, self.state_shape])
        ))
        
        def assert_data_format(x, data_shape, dtype = tf.float32):
            assert x.dtype == dtype
            
            target_shape = tf.concat([batch_shape, data_shape])
            
            tf.assert(
                tf.all(
                    tf.math.logical_or(
                        target_shape == -1,
                        target_shape == tf.shape(x)
                    )
                )
            )
        
        # A batch_shape + [len(universe.types), ?] tensor holding data about all item types
        types = input["types"]
        assert_data_format(types, [self.n_types, -1])
        
        # A batch_shape + [len(universe.systems), ?] tensor holding data about all 
        systems = input["sytems"]
        assert_data_format(system, [self.n_systems, -1])
        
        # A triplet of tensors describing the order data
        (orders_types, orders_systems, orders_data) = input["orders"]
        
        # A tensor of shape batch_shape + [len(orders), ?] holding scalar numeric data about the order
        assert_data_format(orders, [self.n_orders, -1])
        
        # An int32 tensor of shape batch_shape + [len(orders)] holding indices into the second-last dimension of 'types' for type data selection
        assert_data_format(orders_types, [self.n_orders], dtype = tf.int32)
        
        # An int32 tensor of shape batch_shape + [len(orders)] holding indices into the second-last dimension of 'systems' for system data selection
        assert_data_format(orders_systems, [self.n_orders], dtype = tf.int32)
        
        # --- Output shape ---
        
        # Makes a batch_shape + data_shape tensor with a specified default value broadcasted to its shape
        def output_data(data_shape, value, dtype = tf.float32):
            return tf.broadcast_to(tf.constant(value, dtype = dtype), tf.concat([batch_shape, data_shape]))
        
        output = {
            # A tensor holding 3 logits for the actions (move, buy, sell)
            "actions" : output_data([3], 0),
            
            # A tensor holding the logits of a categorical distribution selecting move target systems
            "warp_targets" : output_data([self.n_systems], 0),
            
            # A tensor holding the parameters for the 'buy' action:
            #  [..., 0] holds the logits for a categorical distribution selecting the item to be bought
            #  [..., 1] and [..., 2] hold mean and variance for a normal distribution determining the
            #    amount of items to be bought
            "buy_params" : output_data([self.n_types, 3], [0, 10, 1]),
            
            # A tensor holding the parameters for the 'sell' action, analogously to 'buy_params'
            "sell_params" : output_data([self.n_types, 3], [0, 10, 1]),
            
            # A tensor holding the expected value function (sum of all - potentially time delay discounted - future rewards)
            "value" : output_data([], 0)
        }
        
        # RNN state transfer (output shape = input shape)
        next_state = prev_state
        
        output = output, next_state
    
    def get_initial_state(inputs):
        batch_shape = tf.shape(inputs[1]["types"][:-2])
        
        return tf.zeros(tf.concat([batch_shape, self.state_shape]), dtype = tf.float32)