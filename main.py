import importlib.util
import sys
import random
from skipgram import Skipgram

spec = importlib.util.spec_from_file_location("graph", "graph.py")
graph = importlib.util.module_from_spec(spec)
sys.modules["graph"] = graph
spec.loader.exec_module(graph)

spec = importlib.util.spec_from_file_location("walk", "walks.py")
walks = importlib.util.module_from_spec(spec)
sys.modules["walk"] = walks
spec.loader.exec_module(walks)


def main():
    number_walks = 80
    walk_length = 40
    data_path = "data/data.edgelist"

    G = graph.load_data(data_path)

    print("Number of nodes: {}".format(len(G.nodes())))

    total_num_walks = len(G.nodes()) * number_walks

    data_size = number_walks * walk_length


    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                      path_length=walk_length, alpha=0, rand=random.Random(0))

    representation_size = 128
    window_size = 10

    model = Skipgram(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=1, compute_loss=True)

    output_model = "data/data.embeddings"

    model.save_emb(output_model, len(G.nodes()))

if __name__ == '__main__':
    main()