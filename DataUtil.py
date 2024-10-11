#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import pickle
import sys
import os


sys.setrecursionlimit(10000)
sys.path.append("../")


# copy from CoCoGUM
def save_pickle_data(path_dir, filename, data):
    full_path = path_dir + '/' + filename
    print("Save dataset to: %s" % full_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(full_path, 'wb') as output:
        pickle.dump(data, output)


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)

    return "%02d:%02d:%02d" % (h, m, s)


def make_directory(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    else:
        pass

def read_pickle_data(data_path):
    print("read pickle from %s" % (data_path))
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset