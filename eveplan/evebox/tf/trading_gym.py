import tensorflow as tf

from evebox.actions import buy, sell, warp_to_system
from evebox.tf.embedding import encode_orders, update_encoded_orders

Root = tfp.distributions.JointDistributionsCoroutine.Root


class TradingGym:
    def __init__(self, universe, orders):
        self.universe = universe
        
        self.orders = orders
        self.encoded_orders = encode_orders(orders)
    
    def distribution(self, state, params):
        action_id = yield Root(tfp.distributions.Categorical(logits = params["actions"]))
        
        if action_id == 0:
            target = yield tfp.distributions.Categorical(logits = params["move_targets"])
            target = system_ids[target.numpy()]
            
            return warp_to_system(target)
        
        if action_id == 1:
            order_params = params["buy_params"]
        elif action_id == 2:
            order_params = params["sell_params"]
            
        idx = yield tfp.distributions.Categorical(logits = params[...,0])
        amount = yield tfp.distributions.TruncatedNormal(loc = params[...,1], scale = params[...,2], vmin = 0, vmax = 1e8)[0]
                
        action = buy if action_id == 1 else sell
        
        return action(type_ids[idx.numpy()], amount.numpy())
    
    def sample_with_logp(self, params):
        def inner(params):
            dist = Joint(self.distribution(None, params))
            
            sample = dist.sample()
            logp   = dist.log_prob(sample)
            return sample, logp
        
        return tf.py_function(inner, inp = [params])
    
    def next_state(self, state, params, sample):
        gen = self.distribution(state, params)
        
        try:
            for s in sample:
                gen.end(s)
        except StopIteration as stop:
            return s.value
        
        assert False, "Sample did not match distribution"
            
    
    def encode_state(self, state):
        input = {}
        
        # Encode location
        input['systems'] = tf.constant(
            [
                [
                    1 if state.system == system else 0,
                    universe.distance(state.system, system)
                ]
                for system in universe.systems
            ], dtype = tf.float32
        )
        
        # Encode types
        indices = []
        input['cargo'] = tf.constant(
            [
                [
                    cargo.get(type, (0, 0))[0],
                    cargo.get(type, (0, 0))[1]
                ]
                for cargo in cargos
            ]
        )
        
        input['orders'] = update_encoded_orders(
            self.universe, self.orders, self.updates
        )(self.encoded_orders)
        
    def unroll_model(self, model, tqdm):
        @tf.function
        @tf.recompute_grad
        def _unroll_step(self, prev_state, input):
            next_state, params, value = model((prev_state, input))

            sample, logp = self.sample_with_logp(params)
            return next_state, sample, logp, value
        
        def run(state, n_max):
            model_input = self.encode_state(state)
            model_state = model.get_initial_state(model_input)

            result = []
            
            with tqdm(range(n_max), postfix = state.time_left) as steps:
                for i in steps:
                    next_model_state, sample, logp, value = self._unroll_step(model_state, model_input)

                    result.append()

                    state = self.next_state(state, params, sample)
                    result.append((state, logp, value))
                    
                    steps.set_postfix(state.time_remaining)

                    if state.time_remaining <= 0:
                        break
        
            return states, logps, values
        
        return run
    
    def losses(inputs):
        inputs = list(inputs)
        
        entropy_loss = 0
        value_loss = 0
        policy_loss = 0
        
        for i, (state, logp, value) in enumerate(inputs):
            entropy_loss -= logp
            policy_loss 