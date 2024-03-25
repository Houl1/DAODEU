import os

import numpy as np
import dill
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import csv
import copy
import random
from sklearn.model_selection import train_test_split
from utils.utilities import run_random_walks_n2v
import pandas as pd

# np.random.seed(0)

def load_graphs(dataset_str,time_steps):
    """Load graph snapshots given the name of dataset"""
    graphs = []
    path_list_snapshot = os.listdir("data/{}/{}".format(dataset_str, "/no_repetition_selfloop"))
    path_list_snapshot.sort(key=lambda x:int(x.split('snapshot')[1]))
    for i, file in enumerate(path_list_snapshot):
        if i==time_steps:
            break
        graph = []
        read = open("data/{}/{}/{}".format(dataset_str, "/no_repetition_selfloop", file))
        head = next(read).split(" ")
        for edge in read:
            if edge == "":
                break
            edge = edge.replace('\n','')
            tupl = edge.split(" ")
            tupl = [int(tupl[0]),int(tupl[1])]
            graph.append(tupl)
        G = nx.multigraph.MultiGraph()
        # G = nx.Graph()
        G.add_nodes_from([x for x in range(int(head[0]))])
        G.add_edges_from(graph)
        graphs.append(G)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    # gdvs = []
    # path_list_gdv = os.listdir("data/{}/{}".format(dataset_str, "/gdv"))
    # path_list_gdv.sort(key=lambda x:int(x.split('gdv')[1]))
    # for i,file in enumerate(path_list_gdv):
    #     if i==time_steps:
    #         break
    #     gdv = []
    #     read = open("data/{}/{}/{}".format(dataset_str, "/gdv", file))
    #     for x in read:
    #         tupl = x.split(" ")
    #         tupl = [int(t) for t in tupl]
    #         gdv.append(tupl)
    #     gdv = sp.csr_matrix(gdv, dtype=np.float32)
    #     gdvs.append(gdv)
    # feats = []
    # path_list_feat = os.listdir("data/{}/{}".format(dataset_str, "/feat"))
    # path_list_feat.sort(key=lambda x:int(x.split('feat')[1]))
    # for i,file in enumerate(path_list_feat):
    #     if i==time_steps:
    #         break
    #     feat = []
    #     read = open("data/{}/{}/{}".format(dataset_str, "/feat", file))
    #     for x in read:
    #         tupl = x.split(" ")
    #         tupl = [float(t) for t in tupl[:-1]]
    #         feat.append(tupl)
    #     feat = sp.csr_matrix(feat, dtype=np.float32)
    #     feats.append(feat)
    return graphs, adjs

def get_context_pairs(graphs, adjs):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(len(graphs)):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=20))
    return context_pairs_train

def get_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2, 
                            test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def get_user_evaluation_data(graphs,test_original,test_new):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_user_data_splits(eval_graph, next_graph, test_original,test_new,val_mask_fraction=0.2, 
                            test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def get_shizheng_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_shizheng_data_splits(eval_graph, next_graph, val_mask_fraction=0.2, 
                            test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def get_multistep_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    train_edges_list = list()
    train_edges_false_list = list()
    val_edges_list = list()
    val_edges_false_list = list()
    test_edges_list = list()
    test_edges_false_list = list()
    eval_idx = len(graphs) - 6
    print("Generating eval data ....")
    for i in range(5):
        eval_graph = graphs[eval_idx]
        next_graph = graphs[eval_idx+i+1]

        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2, 
                            test_mask_fraction=0.6)

        train_edges_list.append(train_edges)
        train_edges_false_list.append(train_edges_false)
        val_edges_list.append(val_edges)
        val_edges_false_list.append(val_edges_false)
        test_edges_list.append(test_edges)
        test_edges_false_list.append(test_edges_false)    
    return train_edges_list, train_edges_false_list, val_edges_list, val_edges_false_list, test_edges_list, test_edges_false_list


def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
# def create_data_splits(graph, next_graph, val_mask_fraction=0.3, test_mask_fraction=0.2):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)
    

    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg

def create_user_data_splits(graph, next_graph, test_original,test_new,val_mask_fraction=0.2, test_mask_fraction=0.6):
# def create_data_splits(graph, next_graph, val_mask_fraction=0.3, test_mask_fraction=0.2):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    # for e in edges_next:
    #     if graph.has_node(e[0]) and graph.has_node(e[1]):
    #         edges_positive.append(e)
    edges_positive = np.array(list(zip(test_original,test_new)))
    # edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)
    

    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg

def create_shizheng_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    csv_file = 'data/rocker/new_mapping_file.csv'  # 替换为您的CSV文件路径
    df = pd.read_csv(csv_file,encoding='GBK')

    # 创建一个字典，键为ISO3，值为id集合
    groups = {}
    for _, row in df.iterrows():
        iso3 = row['ISO3']
        id_val = row['ID']
        if iso3 not in groups:
            groups[iso3] = set()
        groups[iso3].add(id_val)

    # 获取所有集合
    sets_list = list(groups.values())

    # 去除重复的集合（即使内部元素顺序不同也认为是重复）
    unique_sets = []
    for s in sets_list:
        if s not in unique_sets:
            unique_sets.append(s)
        
    
    # for e in edges_next:
    #     if graph.has_node(e[0]) and graph.has_node(e[1]):
    #         edges_positive.append(e)
    edges_positive = np.array(unique_sets) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)
    

    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
         
def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

def get_user(graphs,random_numbers,user_percent,time_steps):
    graphs = copy.deepcopy(graphs)
    test_new = [x for x in range(nx.number_of_nodes(graphs[0]),nx.number_of_nodes(graphs[0])+user_percent)]
    correspondence_dict = dict(zip(random_numbers, test_new))
    for i,graph in enumerate(graphs):
        graph.add_nodes_from(test_new)
        # if i>time_steps/2:
        #     for node1 in random_numbers:
        #         neighbors1 = list(graph.neighbors(node1))
        #         corresponding_node2 = correspondence_dict[node1]
        #         for neighbor1 in neighbors1:
        #             graph.add_edge(corresponding_node2, neighbor1)
        #     graph.remove_edges_from([(node, neighbor) for node in random_numbers for neighbor in graph.neighbors(node)])
        # for node1, node2 in correspondence_dict.items():
        #     graph.add_edge(node1, node2)
        for node1 in random_numbers:
            neighbors1 = list(graph.neighbors(node1))
            corresponding_node2 = correspondence_dict[node1]
            for neighbor1 in neighbors1:
                p1 = random.random()
                if p1 <= 0.3:
                    graph.add_edge(corresponding_node2, neighbor1)
                    graph.remove_edge(node1,neighbor1)
                elif p1 > 0.3 and p1 <= 0.5:
                    graph.add_edge(corresponding_node2, neighbor1)
        for node1, node2 in correspondence_dict.items():
            if random.random() <= 0.7:
                graph.add_edge(node1, node2)
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    return  graphs, adjs, random_numbers,test_new