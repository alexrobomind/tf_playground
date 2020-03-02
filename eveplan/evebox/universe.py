import networkx as nx
import multiprocessing.pool

import json
import functools

import evebox.esi as esi

from evebox.util import notqdm

def handle_request(op):
    return json.dumps(esi.request(op).data)

def esi_data_by_ids(name, ids, tqdm = notqdm):
    def get_op(id):
        kwargs = {
            name + '_id' : id
        }
        
        return esi.op['get_universe_{n}s_{n}_id'.format(n = name)](**kwargs)
            
    ops = [get_op(id) for id in ids]
    
    pool = multiprocessing.pool.ThreadPool(5)
            
    d = {
        id : json.loads(data)
        for id, data in zip(
            ids,
            list(tqdm(pool.imap(handle_request, ops, chunksize = 1), desc = 'Loading {}s'.format(name), total = len(ops)))
        )
    }
    
    return d

def esi_data(name, tqdm = notqdm):
    ids = esi.request(esi.op['get_universe_{}s'.format(name)]()).data
    ids = sorted(set(ids))
    return esi_data_by_ids(name, ids, tqdm)

def type_ids(tqdm = notqdm):
    pages = esi.request(
        esi.op['get_universe_types']()
    ).header['X-pages'][0]

    type_ids = [
        i
        for page in tqdm(range(1, pages + 1))
        for i in esi.request(
            esi.op['get_universe_types'](page = page)
        ).data
    ]
    
    type_ids = sorted(set(type_ids))
    
    return type_ids

class Universe:
    def __init__(self):
        self._distance_cache = {}
    
    def distance(self, a, b):
        """
        Computes (and caches) the distance between two systems.
        """
        
        if a in self._distance_cache:
            return self._distance_cache[a][b]
        if b in self._distance_cache:
            return self._distance_cache[b][a]
        
        self._distance_cache[a] = nx.shortest_path_length(self.graph, a)
        
        return self._distance_cache[a][b]
    
    def range(self, root, min_sec = -10.0):
        """
        Constructs a new universe consisting only of the systems reachable from root
        by passing through systems of security status >= min_sec (default min_sec = -10.0)
        """
        
        if root in self.systems_by_name:
            root = self.systems_by_name[root]
            
        assert self.systems[root]["security_status"] >= min_sec, 'The chosen root does not fullfill the security status requirement'
            
        ids = [k for k in self.systems if self.systems[k]["security_status"] >= min_sec]
        
        subgraph = self.graph.subgraph(ids)

        # Limit to components connected to root
        ids = nx.node_connected_component(subgraph, root)
        subgraph = subgraph.subgraph(ids)
        
        u = Universe()
        u.systems = {
            k : v
            for k, v in self.systems.items()
            if k in ids
        }
        
        u.stargates = {
            k : v
            for k, v in self.stargates.items()
            if v["system_id"] in ids
        }
        
        cids = set([v["constellation_id"] for v in self.systems.values()])
        u.constellations = {
            k : v
            for k, v in self.constellations.items()
            if k in cids
        }
        
        u.market_types = self.market_types
        
        return u
    
    def get_region(self, system):
        cid = self.systems[system]["constellation_id"]
        return self.constellations[cid]["region_id"]
    
    @property
    @functools.lru_cache()
    def regions(self):
        regions = [self.get_region(s) for s in self.systems]
        regions = sorted(set(regions))
        return regions
    
    @property
    @functools.lru_cache()
    def systems_by_name(self):
        return {
            v["name"] : k
            for k, v in self.systems.items()
        }
            
    @property
    @functools.lru_cache()
    def graph(self):
        g = nx.Graph()
        
        for system in self.systems:
            g.add_node(system);

        for stargate in self.stargates.values():
            if stargate["destination"]["system_id"] in g:
                g.add_edge(stargate["system_id"], stargate["destination"]["system_id"])
        
        return g
    
    @property
    @functools.lru_cache()
    def system_ids(self):
        return [s for s in self.systems]
        
    def to_json(self, f, *args, **kwargs):
        def do_dump(f):
            json.dump(
                {
                    'systems' : self.systems,
                    'stargates' : self.stargates,
                    'constellations' : self.constellations,
                    'market_types' : self.market_types
                }, f, *args, **kwargs
            )
            
        if type(f) == str:
            with open(f, 'w') as g:
                do_dump(g)
        else:
            do_dump(f)

    @staticmethod
    def from_json(f, *args, **kwargs):
        def do_load(f):
            return json.load(f, *args, **kwargs)
        
        if type(f) == str:
            with open(f, 'r') as g:
                data = do_load(g)
        else:
            data = do_load(f)
            
        def intd(x):
            return {int(k) : v for k,v in x.items()}
        
        u = Universe()
        u.systems = intd(data['systems'])
        u.stargates = intd(data['stargates'])
        u.constellations = intd(data['constellations'])
        u.market_types  = intd(data['market_types'])
        
        return u
    
    @staticmethod
    def from_esi(cache = None, tqdm = notqdm):
        try:
            return Universe.from_json(cache)
        except Exception as e:
            print(e)
            pass
        
        u = Universe()        
        u.systems =        esi_data('system', tqdm)
        u.constellations = esi_data('constellation', tqdm)
        
        tids             = type_ids(tqdm)
        market_types     = esi_data_by_ids('type', tids, tqdm)
        u.market_types   = {
            k : v
            for k, v in market_types.items()
            if 'market_group_id' in v
        }
        del market_types
        
        stargate_ids = sorted(set(
            gate
            for s in u.systems.values()
            for gate in s.get("stargates", [])
        ))
        u.stargates = esi_data_by_ids('stargate', stargate_ids, tqdm)
        
        try:
            u.to_json(cache)
        except Exception as e:
            pass
        
        return u