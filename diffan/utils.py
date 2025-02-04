import numpy as np
import uuid
import pandas as pd
import os


def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i + 1:], order[i]].sum()
    return err


def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i + 1:]] = 1
    return A


def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    # output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
    output = os.path.join(save_path, 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def get_value_from_str(exp_name: str, variable: str, type_func=str):
    value_init_pos = exp_name.rfind(variable) + len(variable)
    if exp_name.rfind(variable) == -1:
        return np.nan
    new_str = exp_name[value_init_pos:]
    end_pos = new_str.find("_")
    if end_pos == -1:
        return type_func(new_str)
    else:
        return type_func(new_str[:end_pos])


# 加的 数据预处理
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', None)


def dataset_transform(data_path=r"/home/newsgrid/linyy/gflowcausal/datasets/25V_474N_Microwave"):
    '''
    将华为比赛数据集转为标准因果数据集
    return: true_causal_matrix, X
    '''

    # 加载数据集
    ## 历史告警
    alarm_path = f"{data_path}/Alarm.csv"
    alarm_data = pd.read_csv(alarm_path, encoding="utf")
    ## 真实因果图
    dag_path = f"{data_path}/DAG.npy"
    true_causal_matrix = np.load(dag_path)
    ## 设备拓扑图
    # topo_path = f"{data_path}/Topology.npy"
    # topo_matrix = np.load(topo_path)
    # topo_matrix[:, :] = 0  # 暂不考虑设备拓扑
    print(f"Load data done, columns: {alarm_data.columns.values}, shape: {alarm_data.shape}")
    print(alarm_data[:10])

    # 得到事件类型及总数
    event_ids = sorted(alarm_data["alarm_id"].unique())
    num_events = len(event_ids)
    print("event_ids:", event_ids, "num_events:", num_events)

    # 将alarm_id的数值变为onehot向量，例如11->[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    alarm_onehot_df = pd.get_dummies(alarm_data["alarm_id"], prefix="e", columns=["alarm_id"])
    data_onehot = alarm_onehot_df.values
    # print(data_onehot.shape)
    # print(data_onehot[:20])

    # check corr 查看相关性系数
    # print(alarm_onehot_df.corr())

    # 滑动窗口
    win_size = true_causal_matrix.shape[0] - 1
    print("win_size: ", win_size)

    # apply sliding window, 得到的data_view是一个一个窗口的切片，但是转置的
    data_view = np.lib.stride_tricks.sliding_window_view(data_onehot, win_size, axis=0)
    # print("data_view:")
    # print(data_view.shape)
    # print(data_view[:5])

    # sum over the sliding window
    data_final = np.sum(data_view[:, :, 1:], axis=-1)
    # print("data_final[:10]:")
    # print(data_final.shape)
    # print(data_final[:10])

    X = data_final

    return true_causal_matrix, X, num_events


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# 加的，可视化因果图
class MyPlotGraphDAG(object):
    '''
    Visualization for causal discovery learning results.

    Parameters
    ----------
    est_dag: np.ndarray
        The DAG matrix to be estimated.
    true_dag: np.ndarray
        The true DAG matrix.
    show: bool
        Select whether to display pictures.
    save_name: str
        The file name of the image to be saved.
    '''

    def __init__(self, est_dag, true_dag=None, info="", show=True, save_name=None):

        self.info = info
        # self.est_dag = est_dag.copy()
        self.est_dag = est_dag
        self.true_dag = true_dag
        self.show = show
        self.save_name = save_name

        if not isinstance(est_dag, np.ndarray):
            raise TypeError("Input est_dag is not numpy.ndarray!")

        if true_dag is not None and not isinstance(true_dag, np.ndarray):
            raise TypeError("Input true_dag is not numpy.ndarray!")

        if not show and save_name is None:
            raise ValueError('Neither display nor save the picture! ' + \
                             'Please modify the parameter show or save_name.')

        MyPlotGraphDAG._plot_dag(self.info, self.est_dag, self.true_dag, self.show, self.save_name)

    @staticmethod
    def _plot_dag(info, est_dag, true_dag, show=True, save_name=None):
        """
        Plot the estimated DAG and the true DAG.

        Parameters
        ----------
        est_dag: np.ndarray
            The DAG matrix to be estimated.
        true_dag: np.ndarray
            The True DAG matrix.
        show: bool
            Select whether to display pictures.
        save_name: str
            The file name of the image to be saved.
        """

        if isinstance(true_dag, np.ndarray):
            est_dag = np.copy(est_dag)
            true_dag = np.copy(true_dag)

            # trans diagonal element into 0
            for i in range(len(true_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0
                if true_dag[i][i] == 1:
                    true_dag[i][i] = 0

            # set plot size
            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

            ax1.set_title('est_dag')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            ax2.set_title('true_dag')
            map2 = ax2.imshow(true_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map2, ax=ax2)

            ax1.text(-8, -3, info)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()



def visualize_dag(adj_matrix):
    # 创建一个有向图对象
    G = nx.DiGraph()

    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 绘制图形
    pos = nx.spring_layout(G, k=2)  # 使用Spring布局算法
    plt.figure(figsize=(12, 12))  # 创建一个新的图形
    nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_color='black',
                     arrows=True)
    plt.title("Directed Acyclic Graph (DAG)")
    plt.show()
