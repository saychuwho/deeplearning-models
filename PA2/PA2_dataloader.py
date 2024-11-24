import pickle
import os
import random

from emo_utils import *


def get_map(glove_type="50d"):
    glove_file = f"./glove.6b.{glove_type}.txt"
    glove_map_file = f"./glove.6b.{glove_type}-map.pkl"
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
    X, Y = read_csv()

    # make label_to_index
    label_to_index = {i:[] for i in range(5)}
    for i, label in enumerate(Y):
        label_to_index[label].append(i)

    # shuffle lebel_to_index
    for i in range(5):
        random.shuffle(label_to_index[i])

    # pop out test_index & train_index
    test_num_divide = [12, 11, 11, 11, 11]
    train_index = []
    test_index = []
    random.shuffle(test_num_divide)
    
    for i, num in enumerate(test_num_divide):
        test_index += [label_to_index[i].pop() for _ in range(num)]
        train_index += label_to_index[i]

    random.shuffle(train_index)
    random.shuffle(test_index)

    # make train set, test set
    train_sentences, train_labels, test_sentences, test_labels = [], [], [], []
    for train_i in train_index:
        train_sentences.append(X[train_i])
        train_labels.append(Y[train_i])

    for test_i in test_index:
        test_sentences.append(X[test_i])
        test_labels.append(Y[test_i])

    return train_sentences, train_labels, test_sentences, test_labels


def train_set_shuffle(train_sentences, train_labels):
    index_list = [i for i in range(len(train_sentences))]
    random.shuffle(index_list)

    ret_train_sentences = [train_sentences[i] for i in index_list]
    ret_train_labels = [train_labels[i] for i in index_list]

    return ret_train_sentences, ret_train_labels