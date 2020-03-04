import tensorflow as tf

from evebox.actions import
from evebox.tf.embedding import encode_orders, update_encoded_orders

Root = tfp.distributions.JointDistributionsCoroutine.Root

def distribution_coroutine

class TradingGym:
    def __init__(self, universe, orders):
        self.universe = universe
        
        self.orders = orders
        self.encoded_orders = encode_orders(orders)
    
    def distribution(self, state, params):
        n_actions = 3 if tf.size(params["sell_params"]) > 0 else 2
        
        action_id = yield Root(tfp.distributions.Categorical(logits = params["actions"][:n_actions]))
        
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
        
        # No sampling after this line
        if state is None:
            return None
        
        if action_id == 1:
            return buy(type_ids[idx].numpy(), amount.numpy())
        elif action_id == 2:
            return sell([type_id for type_id, _ in state.cargo][idx.numpy()], amount.numpy())
        
        assert False
    
    def sample_with_logp(self, params):
        def inner(params):
            dist = Joint(self.distribution(None, params))
            
            sample = self.distribution.sample()
            logp   = self.distribution.log_prob(sample)
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
        
        output['orders'] = update_encoded_orders(
            self.universe, self.orders, self.updates
        )(self.encoded_orders)
        
    def unroll_model(self, model, tqdm = notqdm)
        @tf.function
        @tf.recompute_grad
        def _unroll_step(self, prev_state, input):
            next_state, params = model((prev_state, input))

            sample, logp = self.sample_with_logp(params)
            return state, sample, logp
        
        def run(state, n_max):
            model_input = self.encode_state(state)
            model_state = model.get_initial_state(model_input)

            logp_tot = 0
            states = []

            for i in tqdm(range(n_max)):
                next_model_state, sample, logp = self._unroll_step(model_state, model_input)

                logp_tot += logp

                state = self.next_state(state, params, sample)
                states.append(state)
        
        return states, reward