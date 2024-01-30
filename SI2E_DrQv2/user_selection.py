from args import args
import pickle as pkl
import numpy as np
import random
from sip import *
import networkx as nx
import sys

acc_cadd_sizes = [100, 50, 20]

with open(f"./Detector/{args.dataset[:3]}_user_news_mapping.pkl", "rb") as file:
    user_news_mapping = pkl.load(file)

def user_selection(acc_list, height):
    num_node = len(acc_list)
    adj_matrix = np.zeros((num_node, num_node))
    for acc_id1, acc1 in enumerate(acc_list):
        news = user_news_mapping[acc1]
        for acc_id2, acc2 in enumerate(acc_list):
            if acc_id1 == acc_id2:
                continue
            adj_matrix[acc_id1, acc_id2] += 1.0

    for id in range(num_node):
        if sum(adj_matrix[id]) == 0:
            adj_matrix[id] += 1.0 / num_node
        else:
            adj_matrix[id] /= sum(adj_matrix[id])

    y = PartitionTree(adj_matrix=adj_matrix)
    x = y.build_encoding_tree(height)

    pes = []
    for nid in range(num_node):
        pes.append(y.path_entropy(nid, 0.0))
    pes = np.array(pes)
    sorted_indices = np.argsort(pes)
    indices_min = np.random.choice(sorted_indices[:300], acc_cadd_sizes[0], replace=False)
    middle_start = (len(pes) - 300) // 2
    indices_mid = np.random.choice(sorted_indices[middle_start: middle_start+300], acc_cadd_sizes[1], replace=False)
    indices_max = np.random.choice(sorted_indices[-300:], acc_cadd_sizes[2], replace=False)

    indices = [list(indices_min), list(indices_mid), list(indices_max)]
    controlled_user = []
    for indice in indices:
        user_list = []
        for nid in indice:
            user_list.append(acc_list[nid])
        controlled_user.append(user_list)
    return controlled_user
   
#     tree_structure, leaves = extract_tree_structure(y, height)

#     new_indices = [[], [], []]
#     masks = np.zeros((3, acc_cadd_size))
#     for i in range(masks.shape[0]):
#         indice = indices[i]
#         for j in range(len(leaves)):
#             if leaves[j] in indice:
#                 masks[i, j] = 1
#                 new_indices[i].append(leaves[j])

#     new_tree_structure = [[], [], []]
#     for i in range(len(new_indices)):
#         base_list = masks[i]
#         layer, new_base_list, start_id = [], [], 0
#         for j in range(len(tree_structure)):
#             for node_num in tree_structure[len(tree_structure) - j - 1]:
#                 node_sum = int(sum(base_list[start_id: start_id + node_num]))
#                 start_id += node_num
#                 if node_sum > 0:
#                     layer.append(node_sum)
#                     new_base_list.append(1)
#             new_tree_structure[i].append(layer)
#             base_list = new_base_list.copy()
#             layer, new_base_list, start_id = [], [], 0

#     # # check
#     # for tree in new_tree_structure:
#     #     print(sum(tree[0]), len(tree[0]), sum(tree[1]), len(tree[1]), sum(tree[2]))
#     # sys.exit()

#     agg_ps = [0.0, 0.0, 0.0]
#     for id, new_indice in enumerate(new_indices):
#         agg_ps[id] = y.community_entropy(new_indice)
#     agg_ps /= sum(agg_ps)
    
#     controlled_user = [[], [], []]
#     for i in range(len(new_indices)):
#         for index in new_indices[i]:
#             controlled_user[i].append(id2user[index])

#     return new_tree_structure, controlled_user, agg_ps

# def extract_tree_structure(pt, height):
#     node_layer = []
#     for nid, node in pt.tree_node.items():
#         if node.parent is None:
#             node_layer.append(nid)
#             break
#     tree_structure, leaves = [], []
#     while height != -1:
#         old_node_layer = node_layer.copy()
#         dim_list = []
#         for nid in old_node_layer:
#             node = pt.tree_node[nid]
#             if node.children is None:
#                 leaves.append(node.ID)
#                 continue
#             dim_list.append(len(node.children))
#             for child in node.children:
#                 node_layer.append(child)
#             node_layer.pop(0)
#         tree_structure.append(dim_list)
#         height -= 1
#     return tree_structure[:-1], leaves
