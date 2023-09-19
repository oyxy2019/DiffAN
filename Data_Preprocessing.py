"""
将数据预处理成：O[t1][v1][n1] = t1时间段，v1事件在n1设备上发生的次数
"""
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)


def data_preprocessing(data_path, time_interval=100):
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

    # 计算最大时间、事件总数、设备总数
    max_time = alarm_data['start_timestamp'].max()
    max_event = alarm_data['alarm_id'].max()
    max_device = alarm_data['device_id'].max()

    Out = np.zeros(((max_time // time_interval) + 1, max_event + 1, max_device + 1))

    # 遍历每一行
    for index, row in alarm_data.iterrows():
        Out[row['start_timestamp'] // time_interval][row['alarm_id']][row['device_id']] += 1

    return Out


DatasetName = "25V_474N_Microwave"
O = data_preprocessing(f"./datasets/{DatasetName}")
print(O[0][4][404])
