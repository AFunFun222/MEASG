import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, cdist
from math import sqrt
from torch_cluster import knn
from multiprocessing import cpu_count, Pool
import time
import DataUtil
import config
import datetime
import gc


def KNN_classify(k, X_set, x):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data x
    """

    distances = [sqrt(np.sum((x_compare - x) ** 2)) for x_compare in X_set]

    nearest = np.argsort(distances)  
    node_index = [i for i in nearest[1:k + 1]]
    topK_x = [X_set[i] for i in nearest[1:k + 1]]  
    return node_index, topK_x


def KNN_weigt(x, topK_x):
    distance = []
    v_1 = x  
    data_2 = topK_x

    for i in range(len(data_2)):
        v_2 = data_2[i]

        combine = np.vstack([v_1, v_2])

        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])
    beata = np.mean(distance)  
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))
    return w


def KNN_attr(data, degree):
    '''
    for KNNgraph
    :param data:
    :return:
    '''
    edge_raw0 = []
    edge_raw1 = []
    edge_fea = []


    if len(data) <= 6:
        adjacency_node_num = len(data)-1
    else:
        adjacency_node_num = 6


    for i in range(len(data)):
        x = data[i]

        data_op1 = (data - x) ** 2  
        distances = np.sum(data_op1, axis=1)  
        nearest = np.argsort(distances)  

        node_index = nearest[1:adjacency_node_num + 1]
        topK_x = data[node_index]
   
        loal_distance = np.squeeze(cdist(topK_x, np.array([x]), metric='euclidean'))
        beata = np.mean(loal_distance)  
        loal_weigt = np.exp((-(np.array(loal_distance)) ** 2) / (2 * (beata ** 2)))

        local_index = np.zeros(adjacency_node_num) + i

        edge_raw0 = np.hstack((edge_raw0, local_index))
        edge_raw1 = np.hstack((edge_raw1, node_index))
        edge_fea = np.hstack((edge_fea, loal_weigt))  

    edge_index = [edge_raw0, edge_raw1]

    return edge_index, edge_fea


# KNNGraph
def Gen_graph(data, degree, subtreeIndex):
    loal_distance = []
    data_list = []

    for i in range(len(data)):  
        graph_feature = data[i]

        node_features = torch.tensor(graph_feature, dtype=torch.float)
        edge_raw0 = []
        edge_raw1 = []
        node_edge = []
        w = []

        if subtreeIndex >= len(degree[i]):  
            node_edge = [[0], [1]]
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            w = np.hstack((w, 0.00001))

            edge_features = torch.tensor(np.array(w), dtype=torch.float)

            graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)
        else:
            for index_edge, edge_tree in enumerate(degree[i][subtreeIndex]):
                for node_index in edge_tree:  

                    loal_distance = np.sqrt(np.sum((graph_feature[index_edge]-graph_feature[node_index])**2))

                    edge_raw0 = np.hstack((edge_raw0, index_edge))
                    edge_raw1 = np.hstack((edge_raw1, node_index))
                    w = np.hstack((w, loal_distance))

                node_edge = [edge_raw0, edge_raw1]

            beata = np.mean(w) 

            loal_weigt = np.exp((-(np.array(w)) ** 2) / (2 * (beata ** 2)))

            edge_index = torch.tensor(np.array(node_edge), dtype=torch.long)
            edge_features = torch.tensor(np.array(loal_weigt), dtype=torch.float)

            graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    return data_list


def KNNGraph(data, degree): 
    graphset = []
    subtreeIndex = 0  

    for a_net_data in data:
        graphset.append(Gen_graph(a_net_data, degree, subtreeIndex))  
        subtreeIndex = subtreeIndex+1

    return graphset


