import networkx as nx
import multiprocessing.pool

import json
import functools

import evebox.esi as esi

def esi_data_by_ids(name, ids, tqdm = None):
    if tqdm is None:
        def tqdm(x, desc = None, total = None):
            return x

    kwargs = {
        name + '_id' : id
    }
    
    def get_op(id):
        return esi.op['get_universe_{n}s_{n}_id'.format(n = name)](**kwargs)
        
    def handle_request(id):
        x = esi.request(get_op(id)).data
        return json.dumps(x)
    
    pool = multiprocessing.pool.ThreadPool(5)
            
    d = {
        id : json.loads(data)
         for id, data in zip(
            ids,
            list(tqdm(pool.imap(handle_request, ids, chunksize = 1), desc = 'Loading {}s'.format(name), total = len(ids)))
        )
    }
    
    return d

def esi_data(name, tqdm = None):
    ids = esi.request(esi.op['get_universe_{}s'.format(name)]()).data
    return universe_data_by_ids(name, ids, tqdm)

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
            
        ids = [k for k in systems if systems[k]["security_status"] >= min_sec]
        subgraph = systems_graph.subgraph(ids)

        # Limit to components connected to root
        ids = nx.node_connected_component(subgraph, root)
        subgraph = systems_graph.subgraph(ids)
        
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
        
        cids = set([v["constellation_id"] for v in self.system.values()])
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
    def systems_by_name(self):
        return {
            v["name"] : k
            for k, v in self.systems.values()
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

    @classmethod
    def from_json(self, f, *args, **kwargs):
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
        
        return self
    
    @classmethod
    def from_esi(self, cache = None, tqdm = None):
        try:
            return Universe.from_json(cache)
        except Exception as e:
            print(e)
            pass
        
        u = Universe()        
        u.systems =        esi_data('system', tqdm)
        u.constellations = esi_data('constellation', tqdm)
        
        market_types     = esi_data('type', tqdm)
        u.market_types   = {
            k : v
            for k, v in market_types
            if 'market_group_id' in v
        }
        del market_types
        
        stargate_ids = list(set(
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