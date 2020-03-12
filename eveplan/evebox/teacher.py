import pandas as pd
import numpy as np
import random

import tensorflow_probability as tfp

from evebox.state import State
from evebox.actions import buy, sell, warp_to_system

def propose_action(state, orders):
    # Step 1: Decide whether we want to buy or sell
    
    if random.random() > 0.5:
        # We wanna try selling something from our cargo ...
        sell_to = orders[
            orders['is_buy_order'] &
            (orders['type_id'].isin([k for k, _ in state.cargo]))
        ].copy()

        # ... but of course only with profit
        cdict = {k: v for k, v in state.cargo}
        sell_to['min_price'] = sell_to['type_id'].apply(lambda tid: cdict[tid][1] / cdict[tid][0])

        sell_to = sell_to[
            sell_to['price'] >= sell_to['min_price']
        ]
        
        # Can we sell in this system ?
        sell_in_system = sell_to[
            sell_to['system_id'] == state.system
        ]
        
        if len(sell_in_system) > 0:
            # Cool, let's do that, just pick a random order we can sell to (that includes a high enough price)
            i_order = random.randrange(len(sell_in_system))
            order = sell_in_system.iloc[i_order]
            t = state.universe.types[order['type_id']]
            
            return sell(
                t, cdict[t['type_id']][0], orders
            )
        
        # Oh well, maybe we can go somewhere where we can sell something
        if len(sell_to) > 0:
            i_order = random.randrange(len(sell_to))
            sys_id = sell_to.iloc[i_order]['system_id']
            
            return warp_to_system(
                state.universe, sys_id
            )
    
    # We skipped selling or could not sell, time to dump something into our cargo bay
    # Let's check the market
    buy_from = orders[
        ~orders['is_buy_order']
    ]
    
    buy_in_system = buy_from[
        buy_from['system_id'] == state.system
    ]
    
    # Can we buy here?
    if len(buy_in_system) > 0:
        # Nice, just pick an order and go
        i_order = random.randrange(len(buy_in_system))
        order = buy_in_system.iloc[i_order]
        t = state.universe.types[order['type_id']]
        amount = order['volume_remain']
        
        return buy(
            t, amount, orders
        )
    
    # Could not buy anything in this system
    # Let's go somewhere where we can
    i_order = random.randrange(len(buy_from))
    order = buy_from.iloc[i_order]
    
    return warp_to_system(
        state.universe, order['system_id']
    )