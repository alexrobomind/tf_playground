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
        
        return all_orders
            
        # Sort
        #all_orders = sorted(all_orders, key = lambda x: x.type_id)
        #print('Found {} orders in {}'.format(len(all_orders), universe.regions[region]['name']))
        #all_orders = itertools.groupby(all_orders, key = lambda x: x.type_id)
        #    
        #orders_dict = {
        #    k : v
        #    for k, v in all_orders
        #    if k in typeids
        #}
        
        #print('Reduced to {} type ids'.format(len(orders_dict)))
        #print('Total no of orders: {}'.format(
        #    sum([len(v) for v in orders_dict.values()])
        #))
        #
        #return orders_dict
        
    #orders = {
    #    k : in_region(k)
    #    for k in tqdm(universe.regions, desc = 'Loading market orders')
    #}
    #
    #flat_orders = [
    #    # Encode each order as a dict of columns
    #    {
    #        k : v
    #        for k,v in order.items()
    #    }
    #    for by_region in tqdm(orders.values())
    #    for by_type in by_region.values()
    #    for order in by_type
    #]
    
    orders = [
        {
            k : v
            for k, v in order.items()
        }
        for region in tqdm(universe.regions, desc = 'Loading market orders')
        for order in in_region(region)
    ]
    
    print('Loaded a total of {} orders'.format(len(orders)))
    
    df = pd.DataFrame(orders)
    df.set_index('order_id', inplace = True)
    df.sort_index(inplace = True)
    
    df = df[
        df['type_id'].isin(universe.types) &
        df['system_id'].isin(universe.systems)
    ]
    
    return df