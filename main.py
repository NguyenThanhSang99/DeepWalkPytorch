import importlib.util
import sys
import random
from word2vecmodel import Word2VecModel
import graph


def main():
    number_walks = 8
    walk_length = 10
    data_path = "data/data.edgelist"

    G = graph.load_data(data_path)

    print("Number of nodes: {}".format(len(G.nodes())))

    total_num_walks = len(G.nodes()) * number_walks

    data_size = number_walks * walk_length


    print("Walking...")
    walks_graph = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                      path_length=walk_length, alpha=0, rand=random.Random(0))

    representation_size = 128
    window_size = 10

    model = Word2VecModel(walks_graph, size=representation_size, window=window_size, iter=10, min_count=0, sg=1, hs=1, workers=4, compute_loss=True)

    output_model = "data/data.embeddings"

    print(model.idx2vec)

    model.save_emb(output_model, G.nodes())

if __name__ == '__main__':
    main()