#!/usr/bin/env python
# coding: utf-8

import evebox as box
import pandas as pd
import tensorflow as tf
import cProfile
#from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange

gpus = tf.config.experimental.list_physical_devices('GPU')

# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print('Found {} gpus'.format(len(gpus)))

uni = box.Universe.from_esi(cache = 'universe.txt', tqdm = tqdm).range('Jita', min_sec = 0.5).with_market_types()

#orders = box.load_orders(uni, tqdm = tqdm)
orders = pd.read_csv('market/current.csv')

# Filter all buy orders which are 50x above the median
median   = orders.groupby('type_id')[['price']].median().rename(columns = {'price' : 'median_price'})

orders = orders[
    (50 * orders.merge(median, on = 'type_id')['median_price'] >= orders['price']) |
    ~orders['is_buy_order']
]

# Filter all orders that can not be handled profitably
max_buy  = orders[ orders['is_buy_order']].groupby('type_id')[['price']].max().rename(columns = {'price' : 'max_buy'})
min_sell = orders[~orders['is_buy_order']].groupby('type_id')[['price']].min().rename(columns = {'price' : 'min_sell'})

orders = orders.merge(max_buy, on = 'type_id').merge(min_sell, on = 'type_id')

orders = orders[
    (
        # Sell orders must be below max_buy
        orders['is_buy_order'] |
        (orders['price'] <= orders['max_buy'])
    ) & 
    (
        # Buy orders must be above min_sell
        ~orders['is_buy_order'] |
        (orders['price'] >= orders['min_sell'])
    )
]

# Reduce universe types to all types appearing in the order list

# Pre-filter types
def check_type(t):
    if t["volume"] > 155000:
        return False
    
    return t['type_id'] in orders['type_id']

# Reduce universe
types = [t for t, v in tqdm(uni.types.items()) if check_type(v)]
print('Reducing from {} to {} types'.format(len(uni.types), len(types)))
uni = uni.with_types(types)

# Reduce orders
orders = orders[(orders['type_id'].isin(uni.types)) & (orders['system_id'].isin(uni.systems))]

# Define model class
class Model(tf.keras.Model):
    def __init__(self, universe, bandwidth = 128, d_notes = 4, layers_start = [(16, 8)] * 2, layers_step = [(16, 8)] * 2):
        super().__init__()
        
        self.bandwidth = bandwidth
        self.d_notes  = d_notes
        
        kw = {
            'use_bias' : True
        }
        
        self.embedding = box.tf.Embedding(universe, d_notes = self.d_notes, tqdm = tqdm)
        
        self.input_transforms = {
            k : tf.keras.layers.Dense(self.bandwidth)
            for k in ['state', 'orders', 'cargo', 'systems']
        }
        
        def otf(n):
            return tf.keras.layers.Dense(n, **kw)
        
        self.output_transforms = {
            'actions' : otf(3),
            'move_targets' : otf(1),
            'buy_params' : otf(3),
            'sell_params' : otf(3),
            'value' : otf(1)
        }
        
        def stack(layer_info):
            return box.tf.StackedTogether(
                tf.keras.Sequential([
                    box.tf.MultiHeadedAttention(self.bandwidth, d_head, n_heads) for d_head, n_heads in layer_info
                ])
            )
        
        self.stack_start = stack(layers_start)
        self.stack_step = stack(layers_step)
        
        self.rnn_cell = tf.keras.layers.GRUCell(self.bandwidth)
            
    
    def _preprocess_input(self, input):
        # Pre-process input (adds "note" variables to types & systems, joins orders with their types & systems)
        input = self.embedding(input)
        
        # Expand state to have the same shape as the other stuff
        input['state'] = tf.expand_dims(input['state'], axis = -2)
        
        # Expand all items into a [...,bandwidth] shape
        input = {
            k : self.input_transforms[k](v)
            for k, v in input.items()
        }
        
        return input
        
    def get_initial_state(self, input):
        input = self._preprocess_input(input)
        
        input = self.stack_start(input)
        
        return [tf.reshape(input['state'], [-1, self.bandwidth])]
    
    def call(self, input):
        input, rnn_state = input
        
        batch_shape = tf.shape(input['state'])[:-1]
        
        input = self._preprocess_input(input)
        
        # Add RNN state to the mix
        input['rnn_state'] = tf.reshape(rnn_state, tf.concat([batch_shape, [1, self.bandwidth]], axis = 0))
        
        # We don't want orders in here, too costly (but probably not present anyway)
        if 'orders' in input:
            del input['orders']
        
        # Apply attention stack
        input = self.stack_step(input)
        
        # Extract rnn state & apply to cell
        rnn_in = tf.reshape(input['rnn_state'], [-1, self.bandwidth])
        _, rnn_state = self.rnn_cell(rnn_in, rnn_state)
        del input['rnn_state']
        
        output = {
            'actions' : tf.squeeze(
                self.output_transforms['actions'](input['state']),
                axis = -2
            ),
            
            'move_targets' : tf.squeeze(
                self.output_transforms['move_targets'](input['systems']),
                axis = -1
            ),
            
            'buy_params'  : self.output_transforms['buy_params'] (input['cargo']),
            'sell_params' : self.output_transforms['sell_params'](input['cargo']),
            
            'value' : tf.squeeze(
                self.output_transforms['value'](input['state']),
                axis = [-2, -1]
            )
        }
        
        return output, rnn_state


# Prepare study
import optuna as tuna
from time import time

gym = box.tf.TradingGym(uni, orders)

jita = [s["system_id"] for s in uni.systems.values() if s["name"] == "Jita"][0]

state = box.MutableState()
state.universe  = uni
state.time_left = 100.0
state.system    = jita
state.wallet    = 1e7
state.collateral_limit = 1e6
state.volume_limit = 1e4

state = box.State(state)

import sys

assert len(sys.argv) > 1, 'Please specify max. training time'
maxtime = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def opt_fun(trial):   
    policy_weight  = trial.suggest_loguniform('policy_weight', 1e-12, 1e-4)
    entropy_weight = trial.suggest_loguniform('entropy_weight', 1e-12, 1e-2)
    value_weight   = trial.suggest_loguniform('value_weight', 1e-12, 1e-2)
    learn_speed    = trial.suggest_loguniform('learn_speed', 1e-3, 1)
    
    tqdm.write('Policy weight:  {:.2e}'.format(policy_weight))
    tqdm.write('Entropy weight: {:.2e}'.format(entropy_weight))
    tqdm.write('Value weight:   {:.2e}'.format(value_weight))
    tqdm.write('Learning speed: {:.2e}'.format(learn_speed))
    
    def make_layer_info(name, nmin, nmax):
        n = trial.suggest_int('n_layers_{}'.format(name), nmin, nmax)
        tqdm.write('n_layers_{}: {}'.format(name, n))
        return [
            (
                trial.suggest_int('d_heads_{}_{}'.format(name, i), 1, 16),
                trial.suggest_int('n_heads_{}_{}'.format(name, i), 1, 16)
            )
            for i in range(n)
        ]
    
    model = Model(
        uni,
        bandwidth    = trial.suggest_int('bandwidth', 8, 128),
        d_notes      = trial.suggest_int('d_notes', 1, 8),
        layers_start = make_layer_info('start', 1, 4),
        layers_step  = make_layer_info('step' , 1, 8)
    )
    
    tqdm.write('Bandwidth: {}'.format(model.bandwidth))
    
    unroller = gym.unroll_model(model, tqdm)
    
    # We have to unroll once to make sure the weights exist
    unroller(state, 2)
    
    loss_sideline = tf.Variable(0, dtype = tf.float32)
    
    def loss():
        p_teacher = 1 - minute * learn_speed
        p_teacher = min(1, max(0, p_teacher))
        
        result = unroller(state, 100, p_teacher = p_teacher)
        
        reward = result[-1][0].value - state.value

        policy_loss, value_loss, entropy_loss = gym.losses(result)
        total_loss = value_weight * value_loss + policy_weight * policy_loss + entropy_weight * entropy_loss
        
        tqdm.write('Action 1: {}'.format(result[0][1]))
        
        loss_sideline.assign(total_loss)
        
        tf.summary.scalar('Policy loss', policy_loss, step)
        tf.summary.scalar('Value loss', value_loss, step)
        tf.summary.scalar('Entropy loss', entropy_loss, step)
        tf.summary.scalar('Total loss', total_loss, step)
        tf.summary.scalar('Teacher probability', p_teacher, step)
        tf.summary.scalar('Reward', reward, step)

        tqdm.write('Policy loss:  {}'.format(policy_loss))
        tqdm.write('Value loss:   {}'.format(value_loss))
        tqdm.write('Entropy loss: {}'.format(entropy_loss))
        tqdm.write('Total loss:   {}'.format(total_loss))
        tqdm.write('Reward:       {}'.format(reward))
        
        return total_loss
    
    def performance(n = 20):        
        def single_run():
            result = unroller(state, 100)
            return result[-1][0].value - state.value

        perf = sum([single_run() for i in range(n)]) / n
        
        tqdm.write('')
        tqdm.write('Performance: {}'.format(perf))
        tqdm.write('')
        
        return perf

    opt = tf.keras.optimizers.SGD(1.0)

    writer = tf.summary.create_file_writer('logs/{}'.format(trial.number))
    step = 0
    
    with writer.as_default():
        try:
            with trange(0, maxtime, desc = 'Training', leave = False) as minutes:
                for minute in minutes:
                    t1 = time()
                    while(time() < t1 + 60):
                        opt.minimize(loss, model.trainable_variables)

                        tf.debugging.assert_all_finite(loss_sideline, 'Non-finite loss encountered')
                        
                        step += 1

                    # Report every minute for pruning
                    perf = performance(n = 5)
                    trial.report(perf, minute)
                    minutes.set_postfix({'perf' : perf})
                    
                    tf.summary.scalar('Performance', perf, step)
                    
                    # Report statistics about model weights
                    #for var in model.trainable_variables:
                    #    tf.summary.histogram(var.name, var, step)

                    if trial.should_prune():
                        raise tuna.exceptions.TrialPruned()
                
                model.save('models/{}/model')
                
        except (KeyboardInterrupt, tuna.exceptions.TrialPruned) as e:
            raise e
        except Exception as e:
            tqdm.write('')
            tqdm.write('Exception in trial: {}'.format(e))
            tqdm.write('Cancelling trial')
            tqdm.write('')

            trial.set_user_attr('cancelled_because', repr(e))

            return -1e9
        finally:
            minutes.close()

    return performance()

# Open and run study
study = tuna.load_study(
    study_name = 'tuning',
    storage = 'sqlite:///storage.sqlite',
    pruner = tuna.pruners.HyperbandPruner(),
    sampler = tuna.samplers.TPESampler()
)

study.optimize(opt_fun, catch = (Exception,))



