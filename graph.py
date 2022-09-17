import pandas
from six import iterkeys
from collections import defaultdict, Iterable
from six import iterkeys
import random

class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def consistant(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        removed = 0

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1
        
        return self

    def nodes(self):
        return self.keys()

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v:len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    
def load_data(file, undirected=True):
    graph = Graph()

    with open(file) as f:
        for l in f:
            x, y = l.strip().split(',')[:2]
            x = int(x)
            y = int(y)
            graph[x].append(y)
            if undirected:
                graph[y].append(x)

    graph.consistant()
    return graph

# DeepWalk Algorithms
def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
        # Run random walk
        walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks