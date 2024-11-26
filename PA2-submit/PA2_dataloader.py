import pickle
import os
import random

from emo_utils import *


def get_map(glove_type="50d"):
    glove_file = f"./glove.6B.{glove_type}.txt"
    glove_map_file = f"./glove.6B.{glove_type}-map.pkl"
    map = None

    if os.path.isfile(glove_map_file):
        print(f"... get map from {glove_map_file} ...")
        with open(glove_map_file, 'rb') as map_f:
            map = pickle.load(map_f)
    else:
        print(f"... get map from {glove_file} ...")
        _, _, map = read_glove_vecs(glove_file)
        with open(glove_map_file, 'wb') as map_f:
            pickle.dump(map, map_f)
    return map


def dataloader():
    train_set_file = "./train_emoji.csv"
    test_set_file = "./test_emoji.csv"
    train_sentences, train_labels = read_csv(filename=train_set_file)
    test_sentences, test_labels = read_csv(filename=test_set_file)

    return train_sentences, train_labels, test_sentences, test_labels


def train_set_shuffle(train_sentences, train_labels):
    index_list = [i for i in range(len(train_sentences))]
    random.shuffle(index_list)

    ret_train_sentences = [train_sentences[i] for i in index_list]
    ret_train_labels = [train_labels[i] for i in index_list]

    return ret_train_sentences, ret_train_labels