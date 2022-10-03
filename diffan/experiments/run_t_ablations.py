# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
np.set_printoptions(precision=3)
# internal
from diffscm.diffanm import DiffANM
from diffscm.utils import num_errors, get_value_from_str


def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    dataset_folder = Path(data['dataset_folder'])
    datasets_path_list = sorted(list(dataset_folder.glob("*")))
    assert len(datasets_path_list) > 0, "Zero datasets were found, please use 'gen_synth_data.py' for generating the data or check if the path is correct"
    
    results_dict_keys = ("dataset_name", "steps", "ordering_errors")
    results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
    
    for dataset_path in datasets_path_list:
        exp_name = dataset_path.name
        
        seed = get_value_from_str(exp_name, "seed",int)
        nnodes = get_value_from_str(exp_name, "nnodes",int)
        nedges = get_value_from_str(exp_name, "nedges",int)
        sem = get_value_from_str(exp_name, "sem")

        if seed == 0 and nnodes in [10, 20, 50] and nedges == 5 and sem == "gp":
            pass
        else:
            continue
        

        dataset_adj_path = dataset_path /  "adj_matrix.csv"
        train_set_path = dataset_path /  "train_set.csv"
        true_adj = pd.read_csv(dataset_adj_path).to_numpy()
        X = pd.read_csv(train_set_path).to_numpy()
        num_samples, n_nodes = X.shape
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = torch.FloatTensor(X).to(device)
        print(f"n samples {num_samples} and n nodes {n_nodes} \n Exp {exp_name}")
        method_name = "DiffANM_t_ablations"

        num_samples, n_nodes = X.shape

        diffanm = DiffANM(n_nodes)
        diffanm.train_score(X)
        
        ## Ordering with majority vote
        order = diffanm.topological_ordering(X, None)
        order_errors = num_errors(order, true_adj)
        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["steps"].append(-1)
        
        print(f"{exp_name} - Order errors {order_errors}")
        
        for step in range(0, diffanm.n_steps+1):
            order = diffanm.topological_ordering(X, step)
            order_errors = num_errors(order, true_adj)
            results_dict["dataset_name"].append(exp_name)
            results_dict["ordering_errors"].append(order_errors)
            results_dict["steps"].append(step)
            print(f"Step {step}, Num errors {order_errors}")
            
        


    exp_folder_path = exp_results_folder / method_name
    exp_folder_path.mkdir(parents = True, exist_ok = True)

    pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")



if __name__ == "__main__":

    main()