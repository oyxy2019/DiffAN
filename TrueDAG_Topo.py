from collections import defaultdict, deque

import numpy as np
from castle.metrics import MetricsDAG

from diffan.utils import dataset_transform, visualize_dag, MyPlotGraphDAG
from PruningByTopo import threshold
from cdt.metrics import SID

# 读取数据集
true_causal_matrix, X, n_nodes = dataset_transform("./datasets/25V_474N_Microwave")
visualize_dag(true_causal_matrix)
adj_matrix = true_causal_matrix

dag_path = r"C:\Users\oyxy2019\Desktop\DiffAN\npy\25V.npy"
ori_npy = np.load(dag_path)
ori_npy = threshold(ori_npy, 0.25)
mt = MetricsDAG(ori_npy, true_causal_matrix).metrics
mt["sid"] = SID(true_causal_matrix, ori_npy).item()
print(mt)
# MyPlotGraphDAG(ori_npy, true_causal_matrix)




# Convert adjacency matrix to a graph representation (dictionary)
graph = defaultdict(list)
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix[i])):
        if adj_matrix[i][j] == 1:
            graph[i].append(j)

# Determine root nodes (nodes with no incoming edges)
root_nodes = [node for node in graph if all(adj_matrix[i][node] == 0 for i in range(len(adj_matrix)))]
print(f"root_nodes: {root_nodes}")


def topoSort(graph):
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque()
    for node in graph:
        if in_degree[node] == 0:
            queue.append(node)

    topo_sequence = []
    while queue:
        level_nodes = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level_nodes.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        topo_sequence.append(level_nodes)

    return topo_sequence


# Example adjacency matrix representation of the DAG
adjacency_matrix = true_causal_matrix

graph = defaultdict(list)
for i in range(len(adjacency_matrix)):
    for j in range(len(adjacency_matrix[i])):
        if adjacency_matrix[i][j] == 1:
            graph[i].append(j)

topo_sequence = topoSort(graph)
# print(topo_sequence)
formatted_sequence = []
ans_matrix = np.zeros_like(adjacency_matrix)
# print(ans_matrix)
for level in topo_sequence:
    formatted_sequence.append(",".join(str(node) for node in level))

print("___________________")
for i in range(0, len(topo_sequence)-1):
    sub_lst = topo_sequence[i]
    # print(sub_lst)
    for node1 in sub_lst:
        for j in range(i+1, len(topo_sequence)):
            for node2 in topo_sequence[j]:
                # print(f"node1:{node1} node2:{node2}")
                ans_matrix[node1][node2] = 1

print(f"ans_matrix:\n{ans_matrix}")

print("[" + "],[".join(formatted_sequence) + "]")

for i in range(0, len(ori_npy)):
    for j in range(0, len(ori_npy)):
        if ans_matrix[i][j] == 0:
            ori_npy[i][j] = 0
print("剪枝的结果：")
mt = MetricsDAG(ori_npy, true_causal_matrix).metrics
mt["sid"] = SID(true_causal_matrix, ori_npy).item()
print(mt)

MyPlotGraphDAG(ans_matrix, true_causal_matrix)
MyPlotGraphDAG(ori_npy, true_causal_matrix)


