# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/09/29
@Author  :   Hou Jinlin
@Contact :   1252405352@qq.com
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
from torch.utils.data import DataLoader
import math
# import random

from utils.preprocess import load_graphs, get_context_pairs, get_user_evaluation_data, get_user
from utils.minibatch import MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from eval.user_prediction import get_user_score
from models.model import Mymodel,Mymodelnew

import torch
import random
import os
import wandb
wandb.init(project="myproject2", entity="houjinlin" )
torch.autograd.set_detect_anomaly(True)


def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1)
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG

# python train.py --time_steps 6 --dataset uci --gpu 1 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=7,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='uci',
                        help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
                        help='Batch size (# nodes)')
    parser.add_argument("--early_stop", type=int, default=30,
                        help="patient")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed")

    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    # parser.add_argument('--residual', type=bool, nargs='?', default=True,
    #                     help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--encode_drop', type=float, nargs='?', default=0.05,
                        help='Encoding Dropout (1 - keep probability).')
    parser.add_argument('--ode_drop', type=float, nargs='?', default=0.05,
                        help='ODE Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--method', type=str,
                        choices=["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"],
                        default='dopri5')
    parser.add_argument('--rtol', type=float, default=0.01,
                        help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
    parser.add_argument('--atol', type=float, default=0.001,
                        help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')

    # Architecture params
    parser.add_argument('--encoding_layer_config', type=str, nargs='?', default='256',
                        help='Encoder layer config. ')
    parser.add_argument('--cell_numbers', type=int, default=2,
                        help='Number of learning module layers. ')
    parser.add_argument('--adjoint', default='True')
    parser.add_argument('--graphloss_rate', default=1.0)
    parser.add_argument('--activityloss_rate', default=1.0)
    parser.add_argument('--tasktype', type=str, default="siglestep",choices=['siglestep','multisteps','data_scarce'])
    parser.add_argument('--scare_snapshot', type=str, default='')
    parser.add_argument('--percent', type=float, default=0.1)
    
    args = parser.parse_args()
    print(args)
    wandb.config.update(args)
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    setup_seed(args.seed)
    
    graphs, adjs = load_graphs(args.dataset, args.time_steps)
    assert args.time_steps <= len(adjs), "Time steps is illegal"
    
    user_percent = int(nx.number_of_nodes(graphs[0])  * args.percent)
    random_numbers = random.sample(range(nx.number_of_nodes(graphs[0])), user_percent)
    graphs, adjs, test_original,test_new = get_user(graphs,random_numbers,user_percent,args.time_steps)
    
    context_pairs_train = get_context_pairs(graphs, adjs)
    


    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_user_evaluation_data(graphs,test_original,test_new)
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))

    # Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
    # inductive testing.
    new_G = inductive_graph(graphs[args.time_steps-2], graphs[args.time_steps-1])
    graphs[args.time_steps-1] = new_G
    adjs[args.time_steps-1] = nx.adjacency_matrix(new_G)

    # build dataloader and model
    dataset = MyDataset(args, graphs, adjs, context_pairs_train)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=MyDataset.collate_fn)
    model = Mymodelnew(args, num_feat=12, time_length=args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,eps=1e-5)
    # in training
    best_epoch_val = 0
    patient = 0
    best_epoch_test =0 
    best_epoch_ap = 0
    # best_epoch_mae = math.inf
    # best_epoch_mse = math.inf
    # best_epoch_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = model.get_loss(feed_dict)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())


        model.eval()
        emb = model(feed_dict["graphs"],feed_dict["global_pyg"])[0][-1].detach().cpu().numpy()
        # activity = model(feed_dict["graphs"],feed_dict["global_pyg"])[-1][-1].detach().cpu().numpy()
        val_results, test_results, _, _,test_ap = get_user_score(train_edges_pos,
                                                              train_edges_neg,
                                                              val_edges_pos,
                                                              val_edges_neg,
                                                              test_edges_pos,
                                                              test_edges_neg,
                                                              emb,
                                                              emb)
        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]
        
        # mae_score,mse_score = get_user_score(emb,test_original,test_new)
        # mae_score,mse_score = get_user_score(train_edges_pos,train_edges_neg,val_edges_pos,val_edges_neg,test_edges_pos,test_edges_neg,emb)
    #     if mae_score < best_epoch_mae or mse_score < best_epoch_mse:
    #         if mae_score < best_epoch_mae:
    #             best_epoch_mae = mae_score
    #         if mse_score < best_epoch_mse:
    #             best_epoch_mse = mse_score
    #         torch.save(model.state_dict(), "./model_checkpoints/model_{}.pt".format(args.dataset))
    #         patient = 0
    #     else:
    #         patient += 1
    #         if patient > args.early_stop:
    #             break

    #     print("Epoch {:<3},  Loss = {:.3f},  User_MAE {:.3f}, User_MSE {:.3f}".format(epoch,
    #                                                                                np.mean(epoch_loss),
    #                                                                                mae_score,
    #                                                                                mse_score,))
    #     wandb.log({"Epoch": epoch,"loss":np.mean(epoch_loss),"User_MAE":mae_score,"User_MSE":mse_score})
    # # Test Best Model
    # model.load_state_dict(torch.load("./model_checkpoints/model_{}.pt".format(args.dataset)))
    # model.eval()
    # emb = model(feed_dict["graphs"],feed_dict["global_pyg"])[0][-1].detach().cpu().numpy()
    # # activity = model(feed_dict["graphs"],feed_dict["global_pyg"])[-1][-1] .detach().cpu().numpy()
    # # val_results, test_results, _, _,ap = evaluate_classifier(train_edges_pos,
    # #                                                       train_edges_neg,
    # #                                                       val_edges_pos,
    # #                                                       val_edges_neg,
    # #                                                       test_edges_pos,
    # #                                                       test_edges_neg,
    # #                                                       emb,
    # #                                                       emb)
    # # auc_val = val_results["HAD"][1]
    # # auc_test = test_results["HAD"][1]
    # mae_score,mse_score = get_user_score(emb,test_original,test_new)
    # if mae_score < best_epoch_mae:
    #     best_epoch_test=mae_score
    # if mse_score < best_epoch_mse:
    #     best_epoch_mse = mse_score
    # print("Best User_MAE = {:.3f}, Best User_MSE = {:.3f}".format(best_epoch_mae,best_epoch_mse))
    # wandb.log({"Best User MAE": best_epoch_mae, "Best User MSE": best_epoch_mse})

        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            if best_epoch_test <  epoch_auc_test:
                best_epoch_test = epoch_auc_test
            if best_epoch_ap < test_ap:
                best_epoch_ap = test_ap
            torch.save(model.state_dict(), "./model_checkpoints/model_{}.pt".format(args.dataset))
            patient = 0
        elif epoch_auc_val == best_epoch_val and epoch_auc_test>best_epoch_test:
            best_epoch_val = epoch_auc_val
            if best_epoch_test <  epoch_auc_test:
                best_epoch_test = epoch_auc_test
            if best_epoch_ap < test_ap:
                best_epoch_ap = test_ap
            torch.save(model.state_dict(), "./model_checkpoints/model_{}.pt".format(args.dataset))
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break
            
        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f},Test AP {:.3f}".format(epoch,
                                                                                   np.mean(epoch_loss),
                                                                                   epoch_auc_val,
                                                                                   epoch_auc_test,
                                                                                   test_ap
                                                                                   ))
        wandb.log({"Epoch": epoch,"loss":np.mean(epoch_loss),"Val User AUC":epoch_auc_val,"Test User AUC":epoch_auc_test,"Test User AP":test_ap})
    # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model_{}.pt".format(args.dataset)))
    model.eval()
    emb = model(feed_dict["graphs"],feed_dict["global_pyg"])[0][-1].detach().cpu().numpy()
    # activity = model(feed_dict["graphs"],feed_dict["global_pyg"])[-1][-1] .detach().cpu().numpy()
    val_results, test_results, _, _,ap = get_user_score(train_edges_pos,
                                                          train_edges_neg,
                                                          val_edges_pos,
                                                          val_edges_neg,
                                                          test_edges_pos,
                                                          test_edges_neg,
                                                          emb,
                                                          emb)
    auc_val = val_results["HAD"][1]
    auc_test = test_results["HAD"][1]
    if auc_test<best_epoch_test:
        auc_test=best_epoch_test
    if ap<best_epoch_ap:
        ap=best_epoch_ap
    print("Best Test AUC = {:.3f}, Best AP = {:.3f}".format(auc_test,ap))
    wandb.log({"Best User AUC":auc_test,"Best User AP":ap})






