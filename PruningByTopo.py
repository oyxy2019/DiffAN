import numpy as np
from castle.metrics import MetricsDAG
from cdt.metrics import SID
from diffan.utils import full_DAG, MyPlotGraphDAG, dataset_transform, num_errors


# 定义一个函数，根据条件将方阵中的值设为0或1
def threshold(matrix, value):
    # 创建一个新的方阵，用于存储结果
    result = np.zeros_like(matrix)
    # 遍历方阵中的每个元素
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # 如果元素大于给定的值，将结果方阵中对应位置设为1
            if matrix[i, j] > value:
                result[i, j] = 1
    # 返回结果方阵
    return result


true_causal_matrix, X, n_nodes = dataset_transform("./datasets/25V_474N_Microwave")

if __name__ == '__main__':

    order = [1, 20, 18, 9, 22, 7, 24, 16, 2, 8, 0, 3, 10, 17, 23, 15, 5, 21, 6, 11, 13, 4, 12, 14, 19]

    print("未剪枝的结果：")
    dag_path = r"C:\Users\oyxy2019\Desktop\DiffAN\npy\25V.npy"
    ori_npy = np.load(dag_path)

    # 调用函数，将方阵中小于0.25的值设为0，大于0.25的值设为1
    ori_npy = threshold(ori_npy, 0.25)

    mt = MetricsDAG(ori_npy, true_causal_matrix).metrics
    mt["sid"] = SID(true_causal_matrix, ori_npy).item()
    print(mt)
    MyPlotGraphDAG(ori_npy, true_causal_matrix)

    print("剪枝的结果：")
    order = [1, 20, 18, 9, 22, 7, 24, 16, 2, 8, 0, 3, 10, 17, 23, 15, 5, 21, 6, 11, 13, 4, 12, 14, 19]
    for i in range(len(order)):
        ori_npy[order[i + 1:], order[i]] = 0
    mt = MetricsDAG(ori_npy, true_causal_matrix).metrics
    mt["sid"] = SID(true_causal_matrix, ori_npy).item()
    print(mt)
    MyPlotGraphDAG(ori_npy, true_causal_matrix)
