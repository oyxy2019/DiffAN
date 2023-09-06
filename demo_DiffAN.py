import numpy as np
from castle.datasets import IIDSimulation, DAG
from castle.metrics import MetricsDAG
from diffan.diffan import DiffAN
from diffan.utils import num_errors, dataset_transform, MyPlotGraphDAG
from cdt.metrics import SID
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 改成自己的GPU编号
np.set_printoptions(precision=3)    # 设置numpy小数点精度


def main():
    # data simulation, simulate true causal dag and train_data.
    # n_nodes = 18
    # num_samples = 1000
    # print(f"Creating dataset")
    # weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=4*n_nodes, seed=3)
    # dataset = IIDSimulation(W=weighted_random_dag, n=num_samples, method='nonlinear', sem_type='gp')
    # true_causal_matrix, X = dataset.B, dataset.X

    batch_size = 128
    DatasetName = "25V_474N_Microwave"
    true_causal_matrix, X, n_nodes = dataset_transform(f"./datasets/{DatasetName}")
    X = X.astype(float)
    print(X.shape)
    print(X[:10])

    print(f"Run Causal Discovery with Deciduous Residue")
    diffan = DiffAN(n_nodes, residue=True, batch_size=batch_size, DatasetName=DatasetName)
    adj_matrix, order = diffan.fit(X, use_savemodel="model_state_dict_epoch1221.pth")
    print(f"DiffANM Num errors {num_errors(order, true_causal_matrix)}")
    mt = MetricsDAG(adj_matrix, true_causal_matrix).metrics
    mt["sid"] = SID(true_causal_matrix, adj_matrix).item()
    print(mt)
    MyPlotGraphDAG(adj_matrix, true_causal_matrix)
    print(f"order: {order}")

    # print(f"Run Causal Discovery without Deciduous Residue / Masking only")
    # diffan = DiffAN(n_nodes, residue=False, batch_size=batch_size, DatasetName=DatasetName)
    # adj_matrix, order = diffan.fit(X)
    # print(f"DiffANM Num errors {num_errors(order, true_causal_matrix)}")
    # mt = MetricsDAG(adj_matrix, true_causal_matrix).metrics
    # mt["sid"] = SID(true_causal_matrix, adj_matrix).item()
    # print(mt)
    # MyPlotGraphDAG(adj_matrix, true_causal_matrix)
    # print(f"order: {order}")


if __name__ == "__main__":
    main()


"""
报错记录
1.numpy版本问题，需要大于1.20.0、小于1.23.0

2.报错：
------------------------------------------------------------------------------------------------------------------------
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.20 GiB (GPU 0; 8.00 GiB total capacity; 6.79 GiB already allocated; 0 bytes free; 6.99 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
------------------------------------------------------------------------------------------------------------------------
解决方法：batch_size=1024->256

3.报错：
------------------------------------------------------------------------------------------------------------------------
Loading required package: nlme
This is mgcv 1.9-0. For overview type 'help("mgcv-package")'.
There was some error with gam. The smoothing parameter is set to zero.
Error in smooth.construct.tp.smooth.spec(object, dk$data, dk$knots) : 
  A term has fewer unique covariate combinations than specified maximum degrees of freedom
Calls: pruning ... smooth.construct -> smooth.construct.tp.smooth.spec
Execution halted

R Python Error Output 
------------------------------------------------------------------------------------------------------------------------
解决方法：暂时先不使用CAM剪枝算法，修改了diffan.py的line:62


"""