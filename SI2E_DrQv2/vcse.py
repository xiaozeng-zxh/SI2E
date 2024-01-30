import torch
import pandas as pd
import torch.nn as nn
import utils
import torch.nn.functional as F
import bisect
import math
import numpy as np
from sip import *
import itertools
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RMS(object):
    """
        Online calculation of runtime mean and standard deviation of data
    """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward

    
    
class VCSE(object):
    def __init__(self, knn_k,device):
        self.knn_k = knn_k
        self.device = device

    def __call__(self, state,value):
        #value => [b1 , 1]
        #state => [b1 , c]
        #z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        sim_matrix_s = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)

        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        sim_matrix_v = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        
        sim_matrix = torch.max(torch.cat((sim_matrix_s.unsqueeze(-1),sim_matrix_v.unsqueeze(-1)),dim=-1),dim=-1)[0]
        eps, index = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        state_norm, index = sim_matrix_s.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        value_norm, index = sim_matrix_v.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        eps = eps[:, -1] #k-th nearest distance
        eps = eps.reshape(-1, 1) # (b1, 1)
        
        state_norm = state_norm[:, -1] #k-th nearest distance
        state_norm = state_norm.reshape(-1, 1) # (b1, 1)

        value_norm = value_norm[:, -1] #k-th nearest distance
        value_norm = value_norm.reshape(-1, 1) # (b1, 1)
        
        sim_matrix_v = sim_matrix_v < eps
        n_v = torch.sum(sim_matrix_v,dim=1,keepdim = True) # (b1,1)
        
        sim_matrix_s = sim_matrix_s < eps
        n_s = torch.sum(sim_matrix_s,dim=1,keepdim = True) # (b1,1)        

        reward = torch.special.digamma(n_v+1) / ds + torch.log(eps * 2 + 0.00001)
        return reward, n_v,n_s, eps, state_norm, value_norm
    
class VCSAE(object):
    """
        state entropy estimation based on K-NN considering q-value of each state-action (supervised RL)
    """
    def __init__(self, knn_k, device):
        '''
            knn_k: k parameter
            device: cpu or gpu
        '''
        self.knn_k = knn_k
        self.device = device

    def __call__(self, rep, value):
        '''
            state embedding and their q-values
        '''
        assert rep.shape[0] == value.shape[0]
        num_node = rep.shape[0]
        adj_matrix = np.zeros((num_node, num_node))
        max_value = 0.0
        for nid_i in range(num_node):
            for nid_j in range(nid_i + 1, num_node):
                tmp_value = abs((value[nid_i, 0] - value[nid_j, 0]).cpu().detach().numpy())
                adj_matrix[nid_i, nid_j] += tmp_value
                adj_matrix[nid_j, nid_i] += tmp_value
                max_value = max(max_value, tmp_value)
        for nid_i in range(num_node):
            for nid_j in range(num_node):
                if adj_matrix[nid_i, nid_j] != 0.0:
                    adj_matrix[nid_i, nid_j] = 1.0 - adj_matrix[nid_i, nid_j] / max_value
                    vec_i, vec_j = rep[nid_i, :].cpu().detach().numpy(), rep[nid_j, :].cpu().detach().numpy()
                    cs = (np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) + 1.0) / 2.0
                    adj_matrix[nid_i, nid_j] = max(cs, adj_matrix[nid_i, nid_j])
        y = PartitionTree(adj_matrix=adj_matrix)
        x = y.build_encoding_tree(k=3)
        node_level_0, node_level_1 = list(), list()
        value_level_0, value_level_1 = list(), list()
        length_level_1 = list()
        tmp_set = set()
        for vid, vertex in y.tree_node.items():
            if vertex.children is None:
                node_level_0.append(rep[vid, :])
                value_level_0.append(value[vid, :])
                tmp_set.add(y.tree_node[vertex.parent])
        for vertex in tmp_set:
            tmp_reps, tmp_values, tmp_ens = list(), list(), list()
            for child in vertex.children:
                tmp_reps.append(rep[child, :])
                tmp_values.append(value[child, :])
                tmp_ens.append(y.node_entropy(child))
            assert sum(tmp_ens) >= 0.0
            if sum(tmp_ens) > 0.0:
                tmp_ens /= sum(tmp_ens)
            else:
                assert len(tmp_ens) == 1
                tmp_ens = [1.0]
            nl1, vl1 = tmp_ens[0] * tmp_reps[0], tmp_ens[0] * tmp_values[0]
            for i in range(1, len(tmp_ens)):
                nl1 += tmp_ens[i] * tmp_reps[i]
                vl1 += tmp_ens[i] * tmp_values[i]
            node_level_1.append(nl1)
            value_level_1.append(vl1)
            length_level_1.append(len(tmp_ens))
        node_level_0, node_level_1 = [torch.unsqueeze(tensor, dim=0) for tensor in node_level_0], [torch.unsqueeze(tensor, dim=0) for tensor in node_level_1]
        value_level_0, value_level_1 = [torch.unsqueeze(tensor, dim=0) for tensor in value_level_0], [torch.unsqueeze(tensor, dim=0) for tensor in value_level_1]
        node_level_0, node_level_1 = torch.cat(node_level_0, dim=0), torch.cat(node_level_1, dim=0)
        value_level_0, value_level_1 = torch.cat(value_level_0, dim=0), torch.cat(value_level_1, dim=0)
        reward_0, _, _, _, _, _ = self.entropy_estimate(rep, value)
        reward_1, _, _, _, _, _ = self.entropy_estimate(node_level_1, value_level_1)
        reward_0, reward_1 = reward_0.reshape(-1, 1), reward_1.reshape(-1, 1)
        index = 0
        for i in range(len(reward_1)):
            reward_0[index: index + length_level_1[i]] += (1.0 / length_level_1[i]) * reward_1[i]
            index += length_level_1[i]
        return reward_0

    def entropy_estimate(self, state, value):
        knn_k = min(self.knn_k, state.shape[0])
        #value => [b1 , 1]
        #state => [b1 , c]
        #z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        # similar matrix of state embedding 
        sim_matrix_s = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)

        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        # similar matrix of q-value
        sim_matrix_v = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        
        # max operation
        sim_matrix = torch.max(torch.cat((sim_matrix_s.unsqueeze(-1),sim_matrix_v.unsqueeze(-1)),dim=-1),dim=-1)[0]
        
        # for each state, find its k-th similarity as eps
        eps, index = sim_matrix.topk(knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        state_norm, index = sim_matrix_s.topk(knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        value_norm, index = sim_matrix_v.topk(knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        eps = eps[:, -1]  # k-th nearest distance
        eps = eps.reshape(-1, 1) # (b1, 1)
        
        state_norm = state_norm[:, -1]  # k-th nearest distance for state embedding
        state_norm = state_norm.reshape(-1, 1) # (b1, 1)

        value_norm = value_norm[:, -1]  # k-th nearest distance for q-value
        value_norm = value_norm.reshape(-1, 1) # (b1, 1)
        
        # count the number of eligible neighbors, n_v and n_s
        sim_matrix_v = sim_matrix_v < eps
        n_v = torch.sum(sim_matrix_v,dim=1,keepdim = True) # (b1,1)
        
        sim_matrix_s = sim_matrix_s < eps
        n_s = torch.sum(sim_matrix_s,dim=1,keepdim = True) # (b1,1)        

        reward = torch.special.digamma(n_v+1) / ds + torch.log(eps * 2 + 0.00001)
        return reward, n_v, n_s, eps, state_norm, value_norm

class SMI(object):

    def __init__(self, vs, vt, knn_k, device):
        '''
            knn_k: k parameter
            device: cpu or gpu
        '''
        self.vs = vs
        self.vt = vt
        self.batch_size = vs.shape[0]
        self.y = None
    
    def construct_encoding_tree(self, height=3, min_prob=0.00001, max_prob=0.99999):
        vsa, vta = self.vs.cpu().detach().numpy(), self.vt.cpu().detach().numpy()
        # 计算内积
        inner_product = sigmoid(np.dot(vsa, vta.T))
        # 将内积结果限制在[min_prob, max_prob]范围内
        clipped_inner_product = np.clip(inner_product, min_prob, max_prob)
        # 标准化，使和为1
        normalized_inner_product = clipped_inner_product / np.sum(clipped_inner_product) * 1000.0
        num_node = int(2 * normalized_inner_product.shape[0])
        adj_matrix = np.zeros((num_node, num_node))
        for i in range(normalized_inner_product.shape[0]):
            for j in range(normalized_inner_product.shape[1]):
                if normalized_inner_product[i, j] > 0.05:
                    adj_matrix[i, normalized_inner_product.shape[0] + j] += normalized_inner_product[i, j]
                    adj_matrix[normalized_inner_product.shape[0] + j, i] += normalized_inner_product[i, j]
        self.y = PartitionTree(adj_matrix=adj_matrix)
        x = self.y.build_encoding_tree(height)
        return self.y
    
    def minimize_msi(self):
        # stage 0: nodes of level 0 and 1
        nodes_0, nodes_1 = set(), set()
        tmp_0, tmp_1 = set(), set()
        for nid in range(0, self.batch_size):
            node = self.y.tree_node[nid]
            tmp_0.add(node.parent)
            tmp_1.add(self.y.tree_node[node.parent].parent)
        for nid in range(self.batch_size, 2 * self.batch_size):
            node = self.y.tree_node[nid]
            if node.parent in tmp_0:
                nodes_0.add(node.parent)
            if self.y.tree_node[node.parent].parent in tmp_1:
                nodes_1.add(self.y.tree_node[node.parent].parent)
        # stage 1: paths of minimization and maximization
        min_paths, max_paths = list(), list()
        for nid in nodes_0:
            min_paths.append(nid)
        for nid in nodes_1:
            num = 0
            node = self.y.tree_node[nid]
            for cid in node.children:
                if cid in nodes_0:
                    num += 1
            for i in range(num - 1):
                max_paths.append(nid)
        # stage 2: datasets of postive and negative samples
        dataset_pos, dataset_neg = None, None
        for nid in min_paths:
            datas = self.generate_dataset(nid)
            dataset_pos = torch.unsqueeze(datas[0], dim=0)
            for i in range(1, len(datas)):
                dataset_pos = torch.cat([dataset_pos, torch.unsqueeze(datas[i], dim=0)], dim=0)
        for nid in max_paths:
            datas = self.generate_dataset(nid)
            dataset_neg = torch.unsqueeze(datas[0], dim=0)
            for i in range(1, len(datas)):
                dataset_neg = torch.cat([dataset_neg, torch.unsqueeze(datas[i], dim=0)], dim=0)
        if dataset_neg is None:
            labels = torch.ones([dataset_pos.shape[0], 1]).to(dataset_pos.device)
            return dataset_pos, labels
        elif dataset_pos is None:
            labels = torch.zeros([dataset_neg.shape[0], 1]).to(dataset_neg.device)
            return dataset_neg, labels
        datasets = torch.cat([dataset_pos, dataset_neg], dim=0)
        labels = torch.cat([torch.ones([dataset_pos.shape[0], 1]), torch.zeros([dataset_neg.shape[0], 1])], dim=0).to(datasets.device)
        return datasets, labels

    def maximize_msi(self):
        # stage 0: nodes of level 0 and 1
        nodes_0, nodes_1 = set(), set()
        tmp_0, tmp_1 = set(), set()
        for nid in range(0, self.batch_size):
            node = self.y.tree_node[nid]
            tmp_0.add(node.parent)
            tmp_1.add(self.y.tree_node[node.parent].parent)
        for nid in range(self.batch_size, 2 * self.batch_size):
            node = self.y.tree_node[nid]
            if node.parent in tmp_0:
                nodes_0.add(node.parent)
            if self.y.tree_node[node.parent].parent in tmp_1:
                nodes_1.add(self.y.tree_node[node.parent].parent)
        # stage 1: paths of minimization and maximization
        min_paths, max_paths = list(), list()
        for nid in nodes_0:
            max_paths.append(self.y.tree_node[nid].parent)
        for nid in nodes_1:
            num = 0
            node = self.y.tree_node[nid]
            for cid in node.children:
                if cid in nodes_0:
                    num += 1
            for i in range(num - 1):
                min_paths.append(nid)
        # stage 2: datasets of postive and negative samples
        dataset_pos, dataset_neg = None, None
        for nid in min_paths:
            datas = self.generate_dataset(nid)
            dataset_pos = torch.unsqueeze(datas[0], dim=0)
            for i in range(1, len(datas)):
                dataset_pos = torch.cat([dataset_pos, torch.unsqueeze(datas[i], dim=0)], dim=0)
        for nid in max_paths:
            datas = self.generate_dataset(nid)
            dataset_neg = torch.unsqueeze(datas[0], dim=0)
            for i in range(1, len(datas)):
                dataset_neg = torch.cat([dataset_neg, torch.unsqueeze(datas[i], dim=0)], dim=0)
        if dataset_neg is None:
            labels = torch.ones([dataset_pos.shape[0], 1]).to(dataset_pos.device)
            return dataset_pos, labels
        elif dataset_pos is None:
            labels = torch.zeros([dataset_neg.shape[0], 1]).to(dataset_neg.device)
            return dataset_neg, labels
        datasets = torch.cat([dataset_pos, dataset_neg], dim=0)
        labels = torch.cat([torch.ones([dataset_pos.shape[0], 1]), torch.zeros([dataset_neg.shape[0], 1])], dim=0).to(datasets.device)
        return datasets, labels

    def generate_dataset(self, nid):
        partitions = self.y.tree_node[nid].partition
        data_s, data_t = set(), set()
        for vid in partitions:
            if vid < self.batch_size:
                data_s.add(self.vs[vid])
            elif vid < 2 * self.batch_size:
                data_t.add(self.vt[vid - self.batch_size])
        cp = itertools.product(data_s, data_t)
        datas = [torch.cat([tensor_s, tensor_t], dim=-1) for tensor_s, tensor_t in cp]
        return datas
