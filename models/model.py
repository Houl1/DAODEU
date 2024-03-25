# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/09/27
@Author  :   JinLin Hou
@Contact :   houjinlin@tju.edu.cn
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import scipy.sparse as sp
import numpy as np
import torch_geometric as tg

from models.layers import EncodingLayer, ODEBlock,GODEEncodingLayer,gcnLayer,mindLayer,Encodingfeat,NodeActivityLayer,DynamicsODEBlock,DynamicsNodeActivityLayer,DynamicsDecodingLayer
from utils.utilities import fixed_unigram_candidate_sampler


from utils import *


class Mymodel(nn.Module):
    def __init__(self, args, num_feat, num_gdv, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(Mymodel, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.num_time_steps = time_length-5
        else:
            self.num_time_steps = time_length-1
        self.num_feat = num_feat
        self.num_gdv = num_gdv
        self.num_pr = 1

        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint

        self.encoding, self.ode, self.decoding = self.build_model()
        self.init_params()


    def forward(self, graphs):
        # encoding  forward
        encoding_out = []
        for t in range(self.num_time_steps):
            encoding_out.append(self.encoding(graphs[t]))
        encoding_out = torch.stack([e for e in encoding_out] ).to(encoding_out[0].device)    

        # ODE forward
        ode_out = self.ode(encoding_out)

        #decoding forward
        decoding_out_adj = []
        for ode in ode_out:
            decode_adj = self.decoding(ode)
            decoding_out_adj.append(decode_adj)
        decoding_out_adj = torch.stack([a for a in decoding_out_adj])
        return decoding_out_adj


    def build_model(self):
        # 1:Encoding Layers
        encoding_layer = EncodingLayer(input_dim_feat=self.num_feat,
                                        input_dim_gdv=self.num_gdv,
                                        input_dim_pr=self.num_pr,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # 2:ODE Layers
        ode_layer = ODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)

        #: Decoding Layers
        decoding_layer = DecodingLayer(input_dim = self.encoding_layer_config)
        
        return encoding_layer, ode_layer, decoding_layer



    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()
        
    
    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn

        final_emb = self.forward(graphs)
        self.adj_loss1 = 0.0
        self.graph_loss4 = 0.0


        for t in range(self.num_time_steps):
            
            emb_t = final_emb[t]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            pos_bceloss = nn.BCELoss()
            neg_bceloss = nn.BCELoss()
                                  
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[ :,None , :]*tart_node_neg_emb, dim=2).flatten()
                        
            pos_score = torch.sigmoid(pos_score)
            neg_score = torch.sigmoid(neg_score)
            
            pos_loss = pos_bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = neg_bceloss(neg_score, torch.ones_like(neg_score))

            adj_loss1 = pos_loss + self.args.neg_weight * neg_loss
            self.adj_loss1 += adj_loss1
            
            # MSELoss(t-1≈t)
            L1Loss = nn.L1Loss()
            if t < self.num_time_steps-1:
                graph_loss4 = L1Loss(final_emb[2][t], final_emb[2][t+1])
                self.graph_loss4 += graph_loss4
            
        self.adj_loss1 = self.adj_loss1    
        self.graph_loss4 = self.graph_loss4 * float(self.args.graphloss_rate)                   

        return self.adj_loss1 + self.graph_loss4

class Mymodelnew(nn.Module):
    def __init__(self, args, num_feat, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(Mymodelnew, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.num_time_steps = time_length-5
        else:
            self.num_time_steps = time_length-1
        self.num_feat = num_feat
        self.cell_numbers = args.cell_numbers
        
        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint
        self.encodelayer, self.gcnlayers, self.odelayers, self.node_activity_layers, self.mind_layers = self.build_model()
        # self.encodelayer,self.gcnlayer1, self.ode_layer1, self.gcnlayer2,self.ode_layer2,self.activityLayer,self.mindlayer = self.build_model()
        self.mindlin = nn.Sequential(nn.Linear(self.encoding_layer_config,self.encoding_layer_config,bias=False), nn.Tanh()) 
        # self.init_params()


    def forward(self, graphs,global_pyg):
        # encoding  forward
        feat_out = []
        activities = []
        for t in range(self.num_time_steps):
            graph = copy.deepcopy(graphs[t])
            feat = graph.x
            feat_out.append(self.encodelayer(feat)) 
            act = graph.node_activity
            activities.append(act)
        global_pyg = copy.deepcopy(global_pyg)
        global_edge_index =  global_pyg.edge_index
        global_edge_weight = global_pyg.edge_weight
        # Learning  forward
        names = locals()
        encoding_out_list = list()
        ode_out_list = list()
        activity_out_list = list()
        minding_out_adj_list = list()
        for i in range(self.cell_numbers):
            #gcn
            names['encoding_out'+ str(i)] = []
            for ts in range(self.num_time_steps):
                graph = copy.deepcopy(graphs[ts])
                edge_index = graph.edge_index
                # feat = graph.x
                edge_weight = graph.edge_weight
                if i==0:
                    names['encoding_out'+ str(i)].append(self.gcnlayers[i](feat_out[ts],edge_index,edge_weight))
                    
                else:
                    pass
            if i == 0:
                names['encoding_out'+ str(i)] = torch.stack([e for e in names['encoding_out'+ str(i)]] ).to(names['encoding_out'+ str(i)][0].device)
                encoding_out_list.append(names['encoding_out'+ str(i)])
                    # names['encoding_out'+ str(i)].append(self.gcnlayers[i](names['minding_out_adj' + str(i-1)][t],edge_index,edge_weight))
            # names['encoding_out'+ str(i)] = torch.stack([e for e in names['encoding_out'+ str(i)]] ).to(names['encoding_out'+ str(i)][0].device)                
            
            # ODE forward
            if i==0:
                names['ode_out' + str(i)] = self.odelayers[i](encoding_out_list[i])
                ode_out_list.append(names['ode_out' + str(i)])
                names['activity_out' + str(i)] = self.node_activity_layers[i](ode_out_list[i],activities)
                activity_out_list.append(names['activity_out' + str(i)])
            else:
                names['ode_out' + str(i)] = self.odelayers[i](minding_out_adj_list[i-1])
                ode_out_list.append(names['ode_out' + str(i)])
                names['activity_out' + str(i)] = self.node_activity_layers[i](ode_out_list[i],activity_out_list[i-1])
                activity_out_list.append(names['activity_out' + str(i)])
            
            # mind 
            if i!=self.cell_numbers-1:   
                names['minding_out_adj' + str(i)] = []
                for tm in range(self.num_time_steps):
                    graph = copy.deepcopy(graphs[tm])
                    edge_index = graph.edge_index
                    # feat = graph.x
                    edge_weight = graph.edge_weight
                    mind_adj = self.mind_layers[i](activity_out_list[i][tm],ode_out_list[i][tm],edge_index,edge_weight,global_edge_index,global_edge_weight)
                    names['minding_out_adj' + str(i)].append(mind_adj)
                names['minding_out_adj' + str(i)] = torch.stack([a for a in names['minding_out_adj' + str(i)]])
                minding_out_adj_list.append(names['minding_out_adj' + str(i)])
            
        
        return names['ode_out' + str(self.cell_numbers-1)], names['activity_out' + str(self.cell_numbers-1)]  
            # encoding_out2 = []
            # for t in range(self.num_time_steps+1):
            #     if t == self.num_time_steps:
            #         graph = copy.deepcopy(graphs[t-1])
            #         edge_index = graph.edge_index
            #         edge_weight = graph.edge_weight
            #     else:
            #         graph = copy.deepcopy(graphs[t])
            #         edge_index = graph.edge_index
            #         # feat = graph.x
            #         edge_weight = graph.edge_weight
            #     encoding_out2.append(self.gcnlayer2(minding_out_adj[t],edge_index,edge_weight))
            # # lastode = self.mindlin(minding_out_adj[-1])
            # encoding_out2 = [e for e in encoding_out2]
            # # encoding_out2.append(lastode)
            # encoding_out2 = torch.stack(encoding_out2).to(encoding_out2[0].device)    


            
            # ODE forward
            # ode_out2 = self.ode_layer2(encoding_out2)
            
            # node_activitys = []
            # for ode in ode_out2:
            #     activity = self.activityLayer(ode,graphs[0].max_degs)
            #     node_activitys.append(activity)

        #decoding forward
        # decoding_out_adj = []
        # for ode in ode_out2:
        #     decode_adj = self.decoding_layer(ode)
        #     decoding_out_adj.append(decode_adj)
        # decoding_out_adj = torch.stack([a for a in decoding_out_adj])
        # return ode_out2,node_activitys
        


    def build_model(self):
        # 1:Encoding Layers
        encodelayer = Encodingfeat(input_dim_feat=self.num_feat,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # print(next(encodelayer.parameters()).device)
        # 2:Learning Layers
        gcnlayers = nn.ModuleList()
        for i in range(self.cell_numbers):
            gcnlayer = gcnLayer(input_dim_feat=self.encoding_layer_config,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
            
            gcnlayers.append(gcnlayer)
        # print(next(gcnlayers.parameters()).device)
        # 3:ODE Layers
        odelayers = nn.ModuleList()
        for i in range(self.cell_numbers):
            ode_layer = ODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)
            odelayers.append(module=ode_layer)
        
        # 4:Node Activity Layers
        node_activity_layers = nn.ModuleList()
        for i in range(self.cell_numbers):
            activity_layer = NodeActivityLayer(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)
            node_activity_layers.append(module=activity_layer)    
        
        # 5:mind Layers
        mind_layers = nn.ModuleList()
        for i in range(self.cell_numbers-1):
            mindlayer = mindLayer(input_dim = self.encoding_layer_config,output_dim=self.encoding_layer_config)
            mind_layers.append(module=mindlayer)
        
        # gcnlayer2 = gcnLayer(input_dim_feat=self.encoding_layer_config,
        #                                 output_dim=self.encoding_layer_config,
        #                                 drop=self.encode_drop)
        # ode_layer2 = ODEBlock(self.args,
        #                     encoding_size=self.encoding_layer_config,
        #                     time_steps=self.time_steps,
        #                     dropout=self.ode_drop,
        #                     rtol=self.rtol,
        #                     atol=self.atol,
        #                     method=self.method,
        #                     adjoint =self.adjoint)
        #: Decoding Layers
        # activityLayer = NodeActivityLayer(input_dim = self.encoding_layer_config)
        
        return encodelayer,gcnlayers, odelayers, node_activity_layers, mind_layers



    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()
        
    
    def get_loss(self, feed_dict):
        def s(x):
            return 1 / (1 + np.exp(-x))
        node_1, node_2, node_2_negative, graphs,global_pyg = feed_dict.values()
        # run gnn

        final_emb,node_activity_pre = self.forward(graphs,global_pyg)
        self.adj_loss1 = 0.0
        self.graph_loss4 = 0.0
        self.activity_reconstruction_loss = 0.0
        self.activity_correction_loss = 0.0
        
        activity_recon = []
        for t in range(self.num_time_steps):
            
            emb_t = final_emb[t]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            pos_bceloss = nn.BCELoss()
            neg_bceloss = nn.BCELoss()
            
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[ :,None , :]*tart_node_neg_emb, dim=2).flatten()
                        
            pos_score = torch.sigmoid(pos_score)
            neg_score = torch.sigmoid(neg_score)

            # pos_score = torch.softmax(pos_score)
            # neg_score = torch.softmax(neg_score)
                        
            pos_loss = pos_bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = neg_bceloss(neg_score, torch.ones_like(neg_score))
            
            adj_loss1 = pos_loss + self.args.neg_weight * neg_loss
            self.adj_loss1 += adj_loss1
            
            # MSELoss(t-1≈t)
            # MSELoss = nn.MSELoss()
            L1Loss1 = nn.L1Loss()
            if t < self.num_time_steps-1:
                graph_loss4 = L1Loss1(final_emb[t], final_emb[t+1])
                # graph_loss4 = nn.MSELoss(final_emb[t], final_emb[t+1])
                self.graph_loss4 += graph_loss4
                
            deg_pre = torch.sum(torch.sigmoid(torch.mm(emb_t,emb_t.t())),dim=1)
            activity_recon.append(deg_pre.reshape(deg_pre.shape[0],1))
            
        MSELoss1 = nn.MSELoss()
        activity_pre_matrix = torch.cat([a for a in node_activity_pre[:self.num_time_steps]],dim=1)
        # print(activity_pre_matrix)
        activity_real_matrix = torch.log(torch.cat([a.node_activity for a in graphs[:-1]],dim=1).add(1)) #need
        activity_correction_loss = MSELoss1(activity_pre_matrix,activity_real_matrix)
        self.activity_correction_loss += activity_correction_loss
        
        MSELoss2 = nn.MSELoss()
        # print(activity_recon[0].shape)
        activity_recon_matrix = torch.log(torch.cat([a for a in activity_recon],dim=1).add(1)) #need
        activity_reconstruction_loss = MSELoss2(activity_pre_matrix,activity_recon_matrix)
        self.activity_reconstruction_loss += activity_reconstruction_loss
         
        # self.activity_correction_loss += MSELoss1(activity_real_matrix,activity_recon_matrix) #need
         
        self.adj_loss1 = self.adj_loss1    
        self.graph_loss4 = self.graph_loss4 * float(self.args.graphloss_rate)
        self.activity_correction_loss = self.activity_correction_loss * float(self.args.activityloss_rate)                
        print(self.adj_loss1)
        print(self.graph_loss4)
        print(self.activity_correction_loss)
        print(self.activity_reconstruction_loss)
        print("--------------------------------")
        # return self.adj_loss1
        # return self.adj_loss1 + self.graph_loss4 + self.activity_correction_loss   
        return self.adj_loss1 + self.graph_loss4 + self.activity_correction_loss +  self.activity_reconstruction_loss  

        
class GODE(nn.Module):
    def __init__(self, args, num_feat, num_gdv, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(GODE, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.num_time_steps = time_length-5
        else:
            self.num_time_steps = time_length-1

        self.num_feat = num_feat
        self.num_gdv = num_gdv
        self.num_pr = 1

        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint

        self.encoding, self.ode, self.decoding = self.build_model()
        self.init_params()

        

    # def forward(self, adjs, gdvs, prs):
    def forward(self, graphs):
        # encoding  forward
        encoding_out = []
        for t in range(self.num_time_steps):
            encoding_out.append(self.encoding(graphs[t]))
        encoding_out = torch.stack([e for e in encoding_out] ).to(encoding_out[0].device)    

        # ODE forward
        ode_out = self.ode(encoding_out)

        # decoding forward
        decoding_out_adj = []
        for ode in ode_out:
            decode_adj = self.decoding(ode)
            decoding_out_adj.append(decode_adj)
        decoding_out_adj = torch.stack([a for a in decoding_out_adj])
        return decoding_out_adj

    def build_model(self):
        # 1:Encoding Layers
        encoding_layer = GODEEncodingLayer(input_dim_feat=self.num_feat,
                                        input_dim_gdv=self.num_gdv,
                                        input_dim_pr=self.num_pr,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # 2:ODE Layers
        ode_layer = ODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)

        #: Decoding Layers
        decoding_layer = DecodingLayer(input_dim=self.encoding_layer_config)

        return encoding_layer, ode_layer, decoding_layer

    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()

    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn

        final_emb = self.forward(graphs)
        self.adj_loss1 = 0.0
        self.graph_loss4 = 0.0

        for t in range(self.num_time_steps):

            emb_t = final_emb[t]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            pos_bceloss = nn.BCELoss()
            neg_bceloss = nn.BCELoss()

            pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()

            pos_score = torch.sigmoid(pos_score)
            neg_score = torch.sigmoid(neg_score)

            pos_loss = pos_bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = neg_bceloss(neg_score, torch.ones_like(neg_score))

            adj_loss1 = pos_loss + self.args.neg_weight * neg_loss
            self.adj_loss1 += adj_loss1

            # MSELoss(t-1≈t)
            L1Loss = nn.L1Loss()
            if t < self.num_time_steps - 1:
                graph_loss4 = L1Loss(final_emb[2][t], final_emb[2][t + 1])
                self.graph_loss4 += graph_loss4

        self.adj_loss1 = self.adj_loss1
        self.graph_loss4 = self.graph_loss4 * float(self.args.graphloss_rate)

        return self.adj_loss1 + self.graph_loss4


class DynamicsModel(nn.Module):
    def __init__(self, args, num_feat, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DynamicsModel, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.num_time_steps = time_length-5
        else:
            self.num_time_steps = time_length-1
        self.num_feat = num_feat
        self.cell_numbers = args.cell_numbers
        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint

        self.encodelayer, self.gcnlayers, self.odelayers, self.node_activity_layers, self.mind_layers, self.decoding_layer = self.build_model()
        # self.init_params()


    def forward(self, graph,global_pyg, t):
        # encoding  forward
        feat_out = []
        activities = []
        for i in range(len(t)):
            graph = copy.deepcopy(graph)
            feat = graph.x
            feat_out.append(self.encodelayer(feat)) 
            act = graph.node_activity
            activities.append(act)
        global_pyg = copy.deepcopy(global_pyg)
        global_edge_index =  global_pyg.edge_index
        global_edge_weight = global_pyg.edge_weight
        
        names = locals()
        encoding_out_list = list()
        ode_out_list = list()
        activity_out_list = list()
        minding_out_adj_list = list()
        decoding_out = []
        for i in range(self.cell_numbers):
            #gcn
            names['encoding_out'+ str(i)] = []
            for ts in range(len(t)):
                graph = copy.deepcopy(graph)
                edge_index = graph.edge_index
                # feat = graph.x
                edge_weight = graph.edge_weight
                if i==0:
                    names['encoding_out'+ str(i)].append(self.gcnlayers[i](feat_out[ts],edge_index,edge_weight))
                    
                else:
                    pass
            if i == 0:
                names['encoding_out'+ str(i)] = torch.stack([e for e in names['encoding_out'+ str(i)]] ).to(names['encoding_out'+ str(i)][0].device)
                encoding_out_list.append(names['encoding_out'+ str(i)])
                    # names['encoding_out'+ str(i)].append(self.gcnlayers[i](names['minding_out_adj' + str(i-1)][t],edge_index,edge_weight))
            # names['encoding_out'+ str(i)] = torch.stack([e for e in names['encoding_out'+ str(i)]] ).to(names['encoding_out'+ str(i)][0].device)                
            
            # ODE forward
            if i==0:
                names['ode_out' + str(i)] = self.odelayers[i](encoding_out_list[i],t)
                ode_out_list.append(names['ode_out' + str(i)])
                names['activity_out' + str(i)] = self.node_activity_layers[i](ode_out_list[i],activities,t)
                activity_out_list.append(names['activity_out' + str(i)])
            else:
                names['ode_out' + str(i)] = self.odelayers[i](minding_out_adj_list[i-1],t)
                ode_out_list.append(names['ode_out' + str(i)])
                names['activity_out' + str(i)] = self.node_activity_layers[i](ode_out_list[i],activity_out_list[i-1],t)
                activity_out_list.append(names['activity_out' + str(i)])
            
            # mind 
            if i!=self.cell_numbers-1:   
                names['minding_out_adj' + str(i)] = []
                for tm in range(len(t)):
                    graph = copy.deepcopy(graph)
                    edge_index = graph.edge_index
                    # feat = graph.x
                    edge_weight = graph.edge_weight
                    mind_adj = self.mind_layers[i](activity_out_list[i][tm],ode_out_list[i][tm],edge_index,edge_weight,global_edge_index,global_edge_weight)
                    names['minding_out_adj' + str(i)].append(mind_adj)
                names['minding_out_adj' + str(i)] = torch.stack([a for a in names['minding_out_adj' + str(i)]])
                minding_out_adj_list.append(names['minding_out_adj' + str(i)])
            
            if i==self.cell_numbers-1:
                for ode in names['ode_out' + str(self.cell_numbers-1)]:
                    decode = self.decoding_layer(ode)
                    decoding_out.append(decode)
                decoding_out = torch.stack([a for a in decoding_out])
                
        return decoding_out
        #decoding forward
        decoding_out = []
        for ode in ode_out:
            decode = self.decoding(ode)
            decoding_out.append(decode)
        decoding_out = torch.stack([a for a in decoding_out])
        return decoding_out


    def build_model(self):
        # 1:Encoding Layers
        encodelayer = Encodingfeat(input_dim_feat=self.num_feat,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # 2:Learning Layers
        gcnlayers = nn.ModuleList()
        for i in range(self.cell_numbers):
            gcnlayer = gcnLayer(input_dim_feat=self.encoding_layer_config,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
            
            gcnlayers.append(gcnlayer)
        # 3:ODE Layers
        odelayers = nn.ModuleList()
        for i in range(self.cell_numbers):
            ode_layer = DynamicsODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)
            odelayers.append(module=ode_layer)

        # 4:Node Activity Layers
        node_activity_layers = nn.ModuleList()
        for i in range(self.cell_numbers):
            activity_layer = DynamicsNodeActivityLayer(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)
            node_activity_layers.append(module=activity_layer)    
        
        # 5:mind Layers
        mind_layers = nn.ModuleList()
        for i in range(self.cell_numbers-1):
            mindlayer = mindLayer(input_dim = self.encoding_layer_config,output_dim=self.encoding_layer_config)
            mind_layers.append(module=mindlayer)
        
        decoding_layer = DynamicsDecodingLayer(input_dim = self.encoding_layer_config, output_dim = 1)
        
        return encodelayer,gcnlayers, odelayers, node_activity_layers, mind_layers,decoding_layer



    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()


class GeneDynamics(nn.Module):
    def __init__(self,  A,  b, f=1, h=2):
        super(GeneDynamics, self).__init__()
        self.A = A   # Adjacency matrix
        self.b = b
        self.f = f
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -self.b * (x ** self.f) + torch.sparse.mm(self.A, x**self.h / (x**self.h + 1))
        else:
            f = -self.b * (x ** self.f) + torch.mm(self.A, x ** self.h / (x ** self.h + 1))
        return f

class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self,  L,  k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k   # heat capacity

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
            f = torch.sparse.mm(self.L, x)
        else:
            f = torch.mm(self.L, x)
        return self.k * f


class MutualDynamics(nn.Module):
    #  dx/dt = b +
    def __init__(self, A, b=0.1, k=5., c=1., d=5., e=0.9, h=0.1):
        super(MutualDynamics, self).__init__()
        self.A = A   # Adjacency matrix, symmetric
        self.b = b
        self.k = k
        self.c = c
        self.d = d
        self.e = e
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        n, d = x.shape
        f = self.b + x * (1 - x/self.k) * (x/self.c - 1)
        if d == 1:
            # one 1 dim can be computed by matrix form
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                outer = torch.sparse.mm(self.A,
                                        torch.mm(x, x.t()) / (self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            else:
                outer = torch.mm(self.A,
                                 torch.mm(x, x.t()) / (
                                             self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            f += torch.diag(outer).view(-1, 1)
        else:
            # high dim feature, slow iteration
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                vindex = self.A._indices().t()
                for k in range(self.A._values().__len__()):
                    i = vindex[k, 0]
                    j = vindex[k, 1]
                    aij = self.A._values()[k]
                    f[i] += aij * (x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
            else:
                vindex = self.A.nonzero()
                for index in vindex:
                    i = index[0]
                    j = index[1]
                    f[i] += self.A[i, j]*(x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
        return f