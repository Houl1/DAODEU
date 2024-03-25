from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
from utils.utilities import *
import torch
import torch_sparse
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp
import networkx as nx
import scipy


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args, graphs, adjs, context_pairs):
        super(MyDataset, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.time_steps = args.time_steps-4
        else:
            self.time_steps = args.time_steps

        self.graphs = graphs
        self.adj_original = adjs
        self.adjs = [self._normalize_graph_gcn(a) for a in adjs]
        # self.gdvs = [self._preprocess_gdvs(gdv) for gdv in gdvs]
        self.prs = self.contruct_prs()
        self.degs = self.construct_degs()
        self.feats = self.contruct_feats()
        # self.feats = [self._preprocess_feats(feat) for feat in feats]
        self.node_activities = self.contruct_feats()
        #feats
        self.snis = self.snapshot_nodes_importance()
        # print("snis finish")
        self.gni = self.global_nodes_importance()
        # print("gni finish")
        self.srps = self.structure_relative_position()
        # print("srps finish")
        self.trp = self.time_relatice_position()
        # print("trp finish")
        
        self.context_pairs = context_pairs
        self.max_positive = args.neg_sample_size
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.pyg_graphs = self._build_pyg_graphs()
        self.global_pyg = self.get_global_graph()
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        """
            D^(-0.5)^T * (A+I) * D^(-0.5)
        """
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        
        # return adj
        return adj_normalized


    def _preprocess_gdvs(self, gdv):
        """Row-based normalization of GDV matrix (scipy sparse format). Output is in tuple format"""
        gdv = np.array(gdv.todense())
        return gdv

    def _preprocess_feats(self, feat):
        """Row-based normalization of GDV matrix (scipy sparse format). Output is in tuple format"""
        feat = np.array(feat.todense())
        return feat
    
    def contruct_prs(self):
        """ Compute node pagerank in each graph snapshot."""
        prs = []
        for i in range(self.time_steps):
            G = self.graphs[i]
            pr_dict = nx.pagerank(G)
            pr_list = []
            for j in range(G.number_of_nodes()):
                pr_list.append([pr_dict[j]])
            pr = np.array(pr_list)
            prs.append(pr)
        return prs

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        for i in range(0, self.time_steps):
            G = self.graphs[i]
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs
    
    def contruct_feats(self):
        feats = []
        for i in self.degs:
            x = np.array(i)
            x = x.reshape(-1,1)
            feats.append(x)
        return feats
    
    def snapshot_nodes_importance(self):
        snis = []
        for i_idex,i in enumerate(self.degs):
            x = np.array(i)
            x = x.reshape(-1,1)
            x_two = np.zeros((x.shape[0], 2))
            cl = nx.clustering(nx.DiGraph(self.graphs[i_idex]))
            for j in range(x.shape[0]):
                x_two[j,0] = len(get_neigbors(self.graphs[i_idex],j,2)[2])
                x_two[j,1] = cl[j]
            snis.append(np.concatenate((x,x_two),1))
        return snis
    
    def global_nodes_importance(self):
        global_graph = nx.compose_all(self.graphs[:self.time_steps])
        # print(nx.number_of_nodes(global_graph))
        # print(nx.number_of_edges(global_graph))
        gni = np.zeros((nx.number_of_nodes(global_graph),3)) 
        cl = nx.clustering(nx.DiGraph(global_graph))
        for nodeid in global_graph.nodes():
            out=get_neigbors(global_graph,nodeid,2)
            gni[nodeid,0] = len(out[1])
            gni[nodeid,1] = len(out[2])
            gni[nodeid,2] = cl[nodeid]
        return gni
    
    
    def structure_relative_position(self):
        # global_graph = nx.compose_all(self.graphs[:self.time_steps])
        srps = []
        for adj,deg,graph in zip(self.adj_original,self.degs,self.graphs):
            srp1 = node_sort(deg, graph,list(range(len(deg))))
            # print(srp1)
            srp3 = node_gobal_where(deg,adj,graph)
            # print(srp3)
            srp2 = np.zeros((adj.shape[0],1))
            adj = sp.csr_matrix(adj)
            labels = louvain(adj)
            labels_unique, counts = np.unique(labels, return_counts=True)
            # print(type(labels_unique))
            # print( counts)
            #将社团重新划分子图，把只有一个值的社团直接划分为0。对子图做节点排序
            dic_labels = {}
            for i in range(adj.shape[0]):
                if counts[labels_unique.tolist().index(labels[i])]==1:
                    srp2[i,0] = 0
                else: 
                    if labels[i] in dic_labels:
                        dic_labels[labels[i]].append(i)
                    else:
                        dic_labels[labels[i]]=[]
                        dic_labels[labels[i]].append(i)
            srp2 = node_sort_for_community(dic_labels,graph,srp2)   
            # for i in range(adj.shape[0]):
            #     srp[i,0] = labels_unique[-1]-labels[i]
            #     srp[i,1] = counts[labels[i]]/(labels_unique[-1]+1)
            srp = np.concatenate((srp1,srp2,srp3),1)
            srps.append(srp)
        return srps
            
    def time_relatice_position(self):
        trp = np.zeros((self.adj_original[0].shape[0],3))
        for deg_index,deg in enumerate(self.degs[:self.time_steps]):
            for i_index,i in enumerate(deg):
                if trp[i_index,0]==0 and i>0:
                    trp[i_index,0] =  np.cos((deg_index+1)/(self.time_steps-1))
                    trp[i_index,1] = np.sin((deg_index+1)/(self.time_steps-1))
                    trp[i_index,2] += 1 
                elif trp[i_index,0]>0 and i>0:
                    trp[i_index,1] = np.sin((deg_index+1)/(self.time_steps-1))
                    trp[i_index,2] += 1 
        return trp
                
    def contruct_feats_post(self):
        # Get positional embedding
        pos_emb = None
        for p in range(self.graphs[0].number_of_nodes()):
            if p==0:
                pos_emb = get_pos_emb(p, int(self.args.encoding_layer_config))
            else:
                pos_emb = np.concatenate((pos_emb, get_pos_emb(p, int(self.args.encoding_layer_config))), axis=0)
        return pos_emb

    # def _build_pyg_graphs(self):
    #     pyg_graphs = []
    #     for adj, feat in zip(self.adjs, self.feats):
    #         edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
    #         feat = torch.FloatTensor(feat)
    #         data = Data(x=feat, edge_index=edge_index, edge_weight=edge_weight)
    #         pyg_graphs.append(data)
    #     return pyg_graphs

    def _build_pyg_graphs(self):
        pyg_graphs = []
        max_degs = 0
        for i in self.node_activities[-1]:
            if i.max()>max_degs:
                max_degs = i.max()
        for adj, sni, srp, node_activity in zip(self.adjs, self.snis, self.srps,self.node_activities):
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            feat = np.concatenate((sni,self.gni,srp,self.trp),1)
            feat = torch.FloatTensor(feat)
            node_activity = torch.FloatTensor(node_activity)
            data = Data(x=feat, edge_index=edge_index, edge_weight=edge_weight, node_activity=node_activity,max_degs=max_degs)
            pyg_graphs.append(data)
        return pyg_graphs
    
    def get_global_graph(self):
        global_graph = nx.compose_all(self.graphs[:self.time_steps])
        global_adj = nx.adjacency_matrix(global_graph)
        global_adj = self._normalize_graph_gcn(global_adj)
        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(global_adj)
        global_pyg = Data(edge_index=edge_index, edge_weight=edge_weight)
        return global_pyg
        
    # def _build_pyg_graphs(self):
    #     pyg_graphs = []
    #     num = 0
    #     for adj, sni, srp in zip(self.adjs, self.snis, self.srps):
    #         edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
    #         feat = np.concatenate((sni,self.gni,srp,self.trp),1)
            
    #         np.save('data/'+self.args.dataset+'/feat/feat'+str(num)+'.npy',feat)
    #         # with open('data/'+self.args.dataset+'/feat/feat'+str(num),"w",encoding='UTF-8') as f:
    #         #     for i in r_list:
    #         #         for j in i[0].detach().cpu().numpy():
    #         #             f.write(str(j)+' ')
    #         #         f.write('\n')
    #         num+=1
            
    #         # feat = torch.FloatTensor(feat)
    #         # data = Data(x=feat, edge_index=edge_index, edge_weight=edge_weight)
    #         # pyg_graphs.append(data)
    #     return pyg_graphs
    
    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    def __createitems__(self):
        self.data_items = {}
        for node in list(self.graphs[self.time_steps-1].nodes()):
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []
            for t in range(0, self.time_steps):
                node_1 = []
                node_2 = []
                if len(self.context_pairs[t][node]) > self.max_positive:
                    node_1.extend([node]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][node], self.max_positive, replace=False))
                else:
                    node_1.extend([node]* len(self.context_pairs[t][node]))
                    node_2.extend(self.context_pairs[t][node])
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            node_2_negative = []
            for t in range(len(node_2_list)):
                degree = self.degs[t]
                node_positive = node_2_list[t][:, None]
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree)
                node_2_negative.append(node_negative)
            node_2_neg_list = [torch.LongTensor(np.array(node)) for node in node_2_negative]
            feed_dict['node_1']=node_1_list
            feed_dict['node_2']=node_2_list
            feed_dict['node_2_neg']=node_2_neg_list
            feed_dict["graphs"] = self.pyg_graphs
            feed_dict["global_pyg"]=self.global_pyg

            self.data_items[node] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                concate.append(torch.cat([data[t] for data in data_list]))
            batch_dict[key] = concate
        batch_dict["graphs"] = samples[0]["graphs"]
        batch_dict["global_pyg"] = samples[0]["global_pyg"]
        return batch_dict

    
class DynamicsDataset(Dataset):
    def __init__(self, args, graph, A, feat):
        super(DynamicsDataset, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.time_steps = args.time_steps-4
        else:
            self.time_steps = args.time_steps
        self.graph = graph
        self.A = self._normalize_graph_gcn(A)
        self.feat = feat
        self.train_nodes = list(self.graph.nodes()) # all nodes in the graph.

        self.degs = self.construct_degs()
        self.node_activities = self.contruct_feats()

        self.pyg_graph = self._build_pyg_graph()
        self.global_pyg = self.get_global_graph()
        
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        """
            D^(-0.5)^T * (A+I) * D^(-0.5)
        """
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        for i in range(0, self.time_steps):
            G = self.graph
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs
    
    def contruct_feats(self):
        feats = []
        for i in self.degs:
            x = np.array(i)
            x = x.reshape(-1,1)
            feats.append(x)
        return feats
    
    def get_global_graph(self):
        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(self.A)
        global_pyg = Data(edge_index=edge_index, edge_weight=edge_weight)
        return global_pyg
    
    def _build_pyg_graph(self):
        max_degs = self.node_activities[-1].max()
        node_activity = torch.FloatTensor(self.node_activities[-1])
        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(self.A)
        data = Data(x=self.feat, edge_index=edge_index, edge_weight=edge_weight, node_activity=node_activity,max_degs=max_degs)
        return data

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    def __createitems__(self):
        self.data_items = {}
        for node in list(self.graph.nodes()):
            feed_dict = {}
            feed_dict["graph"] = self.pyg_graph
            feed_dict["global_pyg"] = self.global_pyg
            self.data_items[node] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        batch_dict["graph"] = samples[0]["graph"]
        batch_dict["global_pyg"] = samples[0]["global_pyg"]
        return batch_dict


