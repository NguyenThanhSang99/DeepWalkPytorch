from io import open
from os import path
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

def count_lines(f):
  if path.isfile(f):
    num_lines = sum(1 for line in open(f))
    return num_lines
  else:
    return 0

def write(G, file_path, number_walks, number_workers):
    rand = random.Random(0)
    files_list = ["{}.{}".format(file_path, str(x)) for x in list(range(number_walks))]

    size = len(G)
    files = []

    paths_per_worker = [1 for x in range(number_walks)]

    with ProcessPoolExecutor(max_workers=number_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            files.append(file_)
    return files

class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list
    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()
