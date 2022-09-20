#### Import necessary packages ####
import torch
import torch.nn as nn
import random

def read_data(path):
    f = open(path, 'r', encoding='utf8')
    lines = f.readlines()
    data = [[ int(node) for node in line.rstrip().split(",")] for line in lines if line.rstrip() != '']
    return data

#### Random Walk ####
def RandomWalk(adj_list, started_node, walk_length):
    walk = [started_node]        # Walk starts from this node
    node = started_node
    
    for i in range(walk_length-1):
        random_node = random.randint(0,len(adj_list[node])-1)
        node = adj_list[node][random_node]
        walk.append(node)

    return walk


class Model(torch.nn.Module):
    def __init__(self, size_vertex, embedding_size):
        super(Model, self).__init__()
        self.phi  = nn.Parameter(torch.rand((size_vertex, embedding_size), requires_grad=True))    
        self.phi2 = nn.Parameter(torch.rand((embedding_size, size_vertex), requires_grad=True))
        
        
    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out    = torch.matmul(hidden, self.phi2)
        return out


def skip_gram(model, size_vertex, walk,  window_size, learning_rate):
    for j in range(len(walk)):
        for k in range(max(0,j-window_size) , min(j+window_size, len(walk))):
            #generate one hot vector
            one_hot          = torch.zeros(size_vertex)
            one_hot[walk[j]]  = 1
            
            out              = model(one_hot)
            loss             = torch.log(torch.sum(torch.exp(out))) - out[walk[k]]
            loss.backward()
            
            for param in model.parameters():
                param.data.sub_(learning_rate*param.grad)
                param.grad.data.zero_()

def get_vertex_labels(graph):
    labels = []
    for nodes in graph:
        labels = labels + [node for node in nodes if not node in labels]
    return labels

def DeepWalk(data, embedding_size, window_size, number_walks, walk_length, learning_rate):
    size_vertex = len(data)  # number of vertices
    vertex_labels = get_vertex_labels(data)
    model = Model(size_vertex=size_vertex, embedding_size=embedding_size)
    for i in range(number_walks):
        print("Walk step {}".format(i + 1))
        random.shuffle(vertex_labels)
        for vertex in vertex_labels:
            walk = RandomWalk(data, vertex, walk_length)
            skip_gram(model, size_vertex, walk, window_size, learning_rate)

    return model

def main():
    path = "data/nodes.csv"
    data = read_data(path)
    print(data)

    #### Hyperparameters ####
    window_size  = 3            # window size
    embedding_size  = 5         # embedding size
    number_walks = 200          # walks per vertex
    walk_length  = 6            # walk length
    learning_rate = 0.025       # learning rate

    model = DeepWalk(data, embedding_size, window_size, number_walks, walk_length, learning_rate)

    print(model.phi)

if __name__ == '__main__':
    main()