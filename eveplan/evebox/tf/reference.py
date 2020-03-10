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
        
        self.delta = tf.Variable(0, dtype = tf.float32)
    
    def _check_input(self, input, expect_orders):
        # Check that all required tensors are present
        assert 'state' in input
        assert 'cargo' in input
        assert 'systems' in input
        
        if expect_orders:
            assert 'orders' in input
        
        # Determine batch shape
        batch_shape = tf.shape(input["state"])[:-1]
        
        # Check input shapes
        def assert_data_format(x, data_shape, dtype = tf.float32):
            assert x.dtype == dtype, 'Type mismatch, expected {}, got {}'.format(dtype, x.dtype)
            
            target_shape = tf.concat([batch_shape, data_shape], axis = 0)
            
            tf.debugging.Assert(
                tf.reduce_all(
                    tf.math.logical_or(
                        target_shape == -1,
                        target_shape == tf.shape(x)
                    )
                ),
                ['Invalid data shape', tf.shape(x), batch_shape, data_shape]
            )
        
        # A batch_shape + [?] tensor holding data about the current state
        state = input["state"]
        assert_data_format(state, [-1])
        
        # A batch_shape + [len(universe.types), ?] tensor holding data about all item types
        types = input["cargo"]
        assert_data_format(types, [self.n_types, -1])
        
        # A batch_shape + [len(universe.systems), ?] tensor holding data about all 
        systems = input["systems"]
        assert_data_format(systems, [self.n_systems, -1])
        
        if 'orders' in input:
            # A triplet of tensors describing the order data
            (orders_types, orders_systems, orders_data) = input["orders"]

            # A tensor of shape batch_shape + [len(orders), ?] holding scalar numeric data about the order
            assert_data_format(orders_data, [self.n_orders, -1])

            # An int32 tensor of shape batch_shape + [len(orders)] holding indices into the second-last dimension of 'types' for type data selection
            assert_data_format(orders_types, [self.n_orders], dtype = tf.int32)

            # An int32 tensor of shape batch_shape + [len(orders)] holding indices into the second-last dimension of 'systems' for system data selection
            assert_data_format(orders_systems, [self.n_orders], dtype = tf.int32)
        
        return batch_shape
        
    def call(self, input):
        print('Reference model called')
        print(input)
        # --- Input shape check ---
        
        input, prev_state = input
        
        batch_shape = self._check_input(input, expect_orders = False)
        
        # RNN state shape check
        tf.debugging.Assert(
            tf.reduce_all(
                tf.shape(prev_state) == tf.concat([batch_shape, self.state_shape], axis = 0)
            ),
            ['Invalid state shape', tf.shape(prev_state), batch_shape, self.state_shape]
        )
        
        # --- Output shape ---
        
        # Makes a batch_shape + data_shape tensor with a specified default value broadcasted to its shape
        def output_data(data_shape, value, dtype = tf.float32):
            return tf.broadcast_to(tf.constant(value, dtype = dtype), tf.concat([batch_shape, data_shape], axis = 0))
        
        output = {
            # A tensor holding 3 logits for the actions (move, buy, sell)
            "actions" : output_data([3], 0) + self.delta,
            
            # A tensor holding the logits of a categorical distribution selecting move target systems
            "move_targets" : output_data([self.n_systems], 0),
            
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
        
        return output
    
    def get_initial_state(self, input):
        batch_shape = self._check_input(input, expect_orders = True)
        
        return tf.zeros(tf.concat([batch_shape, self.state_shape], axis = 0), dtype = tf.float32)