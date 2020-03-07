import pandas as pd

import itertools

import evebox.esi as esi

from evebox.universe import Universe
from evebox.util     import notqdm

def load_orders(universe, tqdm = notqdm):
    typeids = set(universe.types)
        
    def in_region(region):
        n_pages = esi.request(
            esi.op['get_markets_region_id_orders'](region_id = region)
        ).header.get('X-pages', [1])[0]
            
        def get_page(i):
            return esi.request(
                esi.op['get_markets_region_id_orders'](region_id = region, page = i, order_type = 'all')
            ).data
            
        all_orders = (
            order
            for i in tqdm(range(1, n_pages + 1), desc = 'Loading orders in ' + universe.regions[region]["name"], leave = False)
            for order in get_page(i)
        )
            
        # Sort
        all_orders = sorted(all_orders, key = lambda x: x.type_id)
        all_orders = itertools.groupby(all_orders, key = lambda x: x.type_id)
            
        return {
            k : v
            for k, v in all_orders
            if k in typeids
        }
        
    orders = {
        k : in_region(k)
        for k in tqdm(universe.regions)
    }
    
    flat_orders = [
        # Encode each order as a dict of columns
        {
            k : v
            for k,v in order.items()
        }
        for by_region in tqdm(orders.values())
        for by_type in by_region.values()
        for order in by_type
    ]
    
    df = pd.DataFrame(flat_orders)
    df.set_index('order_id', inplace = True)
    df.sort_index(inplace = True)
    
    return df