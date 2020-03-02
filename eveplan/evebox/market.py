import evebox.esi as esi

from evebox.universe import Universe
from evebox.util     import notqdm

class Market:
    def __init__(self, universe):
        def unique(x):
            return list(set(x))
        self.universe = universe
        self.regions = unique([universe.region_id(s) for s in universe.systems])
    
    @property
    def types(self):
        return self.universe.market_types
    
    def load_esi(self, tqdm = notqdm):
        def load_orders(region):
            n_pages = esi.request(
                esi.op['']
            )        