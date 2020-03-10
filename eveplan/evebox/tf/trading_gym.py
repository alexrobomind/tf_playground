import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import functools

from evebox.actions import buy, sell, warp_to_system
from evebox.tf.embedding import encode_orders, update_encoded_orders
from evebox.state import MutableState, State

Joint = tfp.distributions.JointDistributionCoroutine

class StructuredGradWrapper():
    def __init__(self, f):
        self.f = f
        
        self.inner = None
        self.example_input = None
        self.example_output = None
        
        self.wrapper = None
    
    def __call__(self, *args, **kwargs):
        if self.inner is None:
            # Extract the structure of the input and the return value
            if self.example_input is None:
                self.example_input = (args, kwargs)
                self.example_input = tf.nest.map_structure(lambda x: 0, self.example_input)
                self.example_output = self.f(*args, **kwargs)
                self.example_output = tf.nest.map_structure(lambda x: 0, self.example_output)
            
            @tf.custom_gradient
            def inner(*input):
                # Unpack input and call f
                (args, kwargs) = tf.nest.pack_sequence_as(self.example_input, input)
                result = self.f(*args, **kwargs)

                # In tensorflow==2.1.0 (and maybe other versions) there is a weird bug where
                # if *dy is present, the 'variables' kwarg is not detected and the grad function
                # therefore refused. However, having a kwargs dict is OK enough to get to
                # variable-supporting mode
                def grad(*dy, **grad_kwargs):
                    variables = list(grad_kwargs.get('variables', []))

                    # For gradient computation, call f again once the gradients
                    # are ready and we can evaluate them swiftly
                    with tf.GradientTape() as t:
                        t.watch(input)
                        t.watch(variables)

                        with tf.control_dependencies([x for x in dy if x is not None]):
                            result = self.f(*args, **kwargs)

                    grad_input, grad_vars = t.gradient(
                        result, (input, variables), output_gradients = dy
                    )

                    return grad_input, grad_vars

                return tf.nest.flatten(result), grad
            
            self.inner = inner

        input = tf.nest.flatten((args, kwargs))
        output = self.inner(*input)
        return tf.nest.pack_sequence_as(self.example_output, output)

def recompute_grad_structured(f):
    return functools.wraps(f)(StructuredGradWrapper(f))


class TradingGym:
    def __init__(self, universe, orders):
        self.universe = universe
        
        self.orders = orders.copy().reset_index().set_index('order_id')
        self.encoded_orders = encode_orders(universe, orders)
        
        self.type_indices = {t: idx for idx, t in enumerate(universe.types)}
        self.system_indices = {s: idx for idx, s in enumerate(universe.systems)}
    
    def sample_and_logp(self, params):
        def move_logits(action):
            action = tf.expand_dims(action, axis = -1)
            result = tf.constant(0, dtype = tf.float32)
            result = tf.where(action == 0, params['move_targets'], result)
            
            return result
        
        def bs_params(action):
            action = tf.expand_dims(action, axis = -1)
            result = tf.constant([0, 0, 1], dtype = tf.float32)
            result = tf.where(action == 1, params['buy_params'], result)
            result = tf.where(action == 2, params['sell_params'], result)
            
            return result
            
        distribution = tfp.distributions.JointDistributionNamed({
            'action' : tfp.distributions.Categorical(
                logits = params['actions']
            ),
            'move_target' : lambda action: tfp.distributions.Categorical(
                logits = move_logits(action)
            ),
            'bs_item' : lambda action: tfp.distributions.Categorical(
                logits = bs_params(action)[..., 0]
            ),
            'bs_amount' : lambda action, bs_item: tfp.distributions.TruncatedNormal(
                loc = tf.abs(bs_params(action)[..., bs_item, 1]),
                scale = tf.abs(bs_params(action)[...,bs_item, 2]),
                low =  0,
                high = 1e8
            )
        })
        
        sample = distribution.sample()
        logp = distribution.log_prob(sample)
        
        return sample, logp
    
    def action(self, sample):
        if sample['action'].numpy() == 0:
            return warp_to_system(self.universe, self.universe.system_list[sample['move_target'].numpy()])
        
        action = buy if sample['action'].numpy() == 1 else sell
        
        return action(self.universe.type_list[sample['bs_item'].numpy()], sample['bs_amount'], self.orders)            
    
    def encode_state(self, state, encode_orders):
        input = {}
        
        # Encode general information
        input['state'] = tf.constant(
            [
                state.time_left
            ],
            dtype = tf.float32
        )
        
        # Encode location
        input['systems'] = tf.constant(
            [
                [
                    1 if state.system == system else 0,
                    self.universe.distance(state.system, system)
                ]
                for system in self.universe.systems
            ], dtype = tf.float32
        )
        
        # Encode types
        cargo_dict = dict(state.cargo)
        input['cargo'] = tf.scatter_nd(
            indices = tf.constant([[self.type_indices[t]] for t in cargo_dict], shape = [len(cargo_dict), 1], dtype = tf.int32),
            updates = tf.constant([val for val in cargo_dict.values()], shape = [len(cargo_dict), 2], dtype = tf.float32),
            shape   = tf.constant([len(self.universe.types), 2], dtype = tf.int32)
        )
        del cargo_dict
        
        if encode_orders:
            updated_ids = np.asarray([k for k,v in state.updated_orders])
            updated_orders = self.orders.loc[updated_ids].copy().sort_index()

            for id, vol in state.updated_orders:
                updated_orders.loc[id]['volume_remain'] = vol

            input['orders'] = update_encoded_orders(
                self.universe, self.orders, updated_orders
            )(self.encoded_orders)
        
        return input
        
    def unroll_model(self, model, tqdm):
        # Since the recompute grad decorator does not handle structured data
        #  (e.g. dicts), we need to flatten data along its boundary. For the
        #  graph-mode decoding, we need to have the structure of these data
        #  ready. Therefore we use an example state to infer the structure
        #  of the model input.
        
        # Creation of example state
        #example_state = MutableState()
        #example_state.time_left = 1.0
        #example_state.system = self.universe.system_list[0]
        
        #example_state = State(example_state)
        
        # Encoding of example state
        #example_input = self.encode_state(example_state)
        #example_model_state = model.get_initial_state(example_input)
        #example_output, _ = model(
        #    (
        #        example_input,
        #        model.get_initial_state(example_input)
        #    )
        #)
        #example_sample, _ = self.sample_and_logp(example_output)
        
        @tf.function
        @recompute_grad_structured
        def unroll_step(prev_state, input):            
            # Compute model output
            model_output, next_state = model((input, prev_state))
            sample, logp = self.sample_and_logp(model_output)
            value = model_output['value']
            
            ## Pack data into flat list
            #result = [logp, value, next_state] + tf.nest.flatten(sample)
            #
            #return result
            return next_state, logp, value, sample
        
        @tf.function
        @recompute_grad_structured
        def initial_value(input):
            return model.get_initial_state(input)
        
        def run(state, n_max):
            model_input = self.encode_state(state, encode_orders = True)
            model_state = initial_value(model_input)

            result = []
            
            with tqdm(range(n_max), postfix = state.time_left) as steps:
                for i in steps:
                    steps.set_postfix({'Time left' : state.time_left})
                    model_input = self.encode_state(state, encode_orders = False)
                    
                    # Handle the bulk of the model in graph mode
                    model_state, logp, value, sample = unroll_step(model_state, model_input)
                    
                    action = self.action(sample)
                    
                    # Record this sequence
                    result.append((state, action, logp, value))

                    # Check for early termination (no point advancing if the time budget is out)
                    if state.time_left <= 0:
                        break
                    
                    # Advance state and update progress bar info
                    state = action(state)
        
            return result
        
        return run
    
    def losses(self, inputs):
        inputs = list(inputs)
        
        states, actions, logps, values = zip(*inputs)
        
        logps = tf.stack(logps, axis = -1)
        values = tf.stack(values, axis = -1)
        state_values = tf.constant(
            [state.value for state in states],
            dtype = tf.float32
        )
        tl = tf.constant(
            [state.time_left for state in states],
            dtype = tf.float32
        )
        
        rewards = state_values[...,1:] - state_values[...,:-1]
        q_values = tf.math.cumsum(rewards, reverse = True)
        advantages = q_values - values[:-1]
        
        policy_loss  = tf.math.reduce_mean(-logps[:-1] * advantages)
        value_loss   = tf.math.reduce_mean(0.5 * advantages**2)
        entropy_loss = tf.math.reduce_mean(logps)
        
        return policy_loss, value_loss, entropy_loss