import numpy as np
from castle.datasets import IIDSimulation, DAG
from castle.metrics import MetricsDAG
from diffan.diffan import DiffAN
from diffan.utils import num_errors, dataset_transform, MyPlotGraphDAG
from cdt.metrics import SID
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=3)


def main():
    # # data simulation, simulate true causal dag and train_data.
    # n_nodes = 10
    # num_samples = 1000
    # print(f"Creating dataset")
    # weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=4*n_nodes, seed=3)
    # dataset = IIDSimulation(W=weighted_random_dag, n=num_samples, method='nonlinear', sem_type='gp')
    # true_causal_matrix, X = dataset.B, dataset.X

    true_causal_matrix, X, n_nodes = dataset_transform("datasets/18V_55N_Wireless")

    print(f"Run Causal Discovery with Deciduous Residue")
    diffan = DiffAN(n_nodes, residue=True)
    adj_matrix, order = diffan.fit(X)
    print(f"DiffANM Num errors {num_errors(order, true_causal_matrix)}")
    mt = MetricsDAG(adj_matrix, true_causal_matrix).metrics
    mt["sid"] = SID(true_causal_matrix, adj_matrix).item()
    print(mt)
    MyPlotGraphDAG(1, adj_matrix, true_causal_matrix)

    print(f"Run Causal Discovery without Deciduous Residue / Masking only")
    diffan = DiffAN(n_nodes, residue=False)
    adj_matrix, order = diffan.fit(X)
    print(f"DiffANM Num errors {num_errors(order, true_causal_matrix)}")
    mt = MetricsDAG(adj_matrix, true_causal_matrix).metrics
    mt["sid"] = SID(true_causal_matrix, adj_matrix).item()
    print(mt)
    MyPlotGraphDAG(2, adj_matrix, true_causal_matrix)


if __name__ == "__main__":
    main()
