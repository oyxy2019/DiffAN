
from logging import raiseExceptions
import torch
import numpy as np
from functorch import vmap, jacrev, jacfwd
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

from diffan.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
                                         LossType, ModelMeanType, ModelVarType
from diffan.nn import DiffMLP
from diffan.pruning import cam_pruning
from diffan.utils import full_DAG


class DiffAN():
    def __init__(self, n_nodes, masking = True, residue= True, 
                epochs: int = int(3e3), batch_size : int = 1024, learning_rate : float = 0.001, DatasetName = None):
        self.n_nodes = n_nodes
        assert self.n_nodes > 1, "Not enough nodes, make sure the dataset contain at least 2 variables (columns)."
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## Diffusion parameters
        self.n_steps = int(1e2)
        betas = get_named_beta_schedule(schedule_name = "linear", num_diffusion_timesteps = self.n_steps, scale = 1, beta_start = 0.0001, beta_end = 0.02)
        self.gaussian_diffusion = GaussianDiffusion(betas = betas, 
                                                    loss_type = LossType.MSE, 
                                                    model_mean_type= ModelMeanType.EPSILON,  # START_X,EPSILON
                                                    model_var_type=ModelVarType.FIXED_LARGE,
                                                    rescale_timesteps = True,
                                                    )
        self.schedule_sampler = UniformSampler(self.gaussian_diffusion)

        ## Diffusion training
        self.epochs = epochs 
        self.batch_size = batch_size
        self.model = DiffMLP(n_nodes).to(self.device)
        self.model.float()
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.val_diffusion_loss = []
        self.best_loss = float("inf")
        self.early_stopping_wait = 300

        ## Topological Ordering
        self.n_votes = 3
        self.masking = masking
        self.residue = residue
        self.sorting = (not masking) and (not residue)

        ## Pruning
        self.cutoff = 0.001

        ## 加的
        self.DatasetName = DatasetName
        self.custom = True

    def fit(self, X, use_savemodel=None):
        # X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)    # 标准化处理
        X = torch.FloatTensor(X).to(self.device)
        self.train_score(X, use_savemodel=use_savemodel)
        order = self.topological_ordering(X)
        out_dag = self.pruning(order, X.detach().cpu().numpy())
        return out_dag, order

    def pruning(self, order, X):
        return full_DAG(order)
        # return cam_pruning(full_DAG(order), X, self.cutoff)
    
    def train_score(self, X, fixed=None, use_savemodel=None):
        if fixed is not None:
            self.epochs = fixed
        best_model_state_epoch = 300
        self.model.train()  # self.model: 一个神经网络模型，用来预测给定噪声图像和时间步的噪声分布
        n_samples = X.shape[0]
        self.batch_size = min(n_samples, self.batch_size)

        # 加载跑过的模型，节约运行时间
        if(use_savemodel is not None):
            print(f"##### using_savemodel:{use_savemodel} #####")
            self.model.load_state_dict(torch.load(f'./save_model/{use_savemodel}'))
            return

        val_ratio = 0.2    # 表示验证集占总数据集的比例
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size   # 表示训练集的大小
        X = X.to(self.device)
        X_train, X_val = X[:train_size],X[train_size:]  # 两个二维数组，表示划分后的训练集和验证集
        data_loader_val = torch.utils.data.DataLoader(X_val, min(val_size, self.batch_size))    # 一个数据加载器，用来从验证集中按批次提取数据
        data_loader = torch.utils.data.DataLoader(X_train, min(train_size, self.batch_size), drop_last= True)   # 一个数据加载器，用来从训练集中按批次提取数据
        pbar = tqdm(range(self.epochs), desc = "Training Epoch")    # 一个进度条，用来显示训练轮数和验证损失
        for epoch in pbar:  # 使用一个循环来进行训练
            loss_per_step = []  # 一个列表，用来存放每个时间步的训练损失
            for steps, x_start in enumerate(data_loader):   # 两个变量，表示当前的批次编号和批次数据。使用一个内部循环来遍历训练集中的每个批次数据
                # apply noising and masking
                x_start = x_start.float().to(self.device)
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)    # 两个一维数组，表示从采样器中抽取的时间步和对应的权重
                noise = torch.randn_like(x_start).to(self.device)
                x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)    # 对批次数据进行噪声化和掩码处理，得到噪声图像x_t
                # get loss function
                model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))    # 使用模型对噪声图像进行预测，得到模型输出model_output
                diffusion_losses = (noise - model_output) ** 2  # 计算模型输出和真实噪声之间的均方误差作为损失函数diffusion_loss
                diffusion_loss = (diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                loss_per_step.append(diffusion_loss.item())    # 将损失函数加入到一个列表loss_per_step中，用来记录每个时间步的训练损失
                self.opt.zero_grad()
                diffusion_loss.backward()
                self.opt.step()    # 使用优化器对模型参数进行更新
            if fixed is None:
                if epoch % 10 == 0 and epoch > best_model_state_epoch:  # 如果没有指定固定的训练轮数，并且当前轮数是10的倍数，并且当前轮数大于最佳模型状态所在的轮数
                    with torch.no_grad():
                        loss_per_step_val = []
                        for steps, x_start in enumerate(data_loader_val):   # 使用一个内部循环来遍历验证集中的每个批次数据
                            t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                            noise = torch.randn_like(x_start).to(self.device)
                            x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)    # 对批次数据进行噪声化和掩码处理，得到噪声图像
                            model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))    # 使用模型对噪声图像进行预测，得到模型输出
                            diffusion_losses = (noise - model_output) ** 2
                            diffusion_loss = (diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()     # 计算模型输出和真实噪声之间的均方误差作为损失函数
                            loss_per_step_val.append(diffusion_loss.item())    # 将损失函数加入到一个列表loss_per_step_val中，用来记录每个时间步的验证损失
                        epoch_val_loss = np.mean(loss_per_step_val)

                        if self.best_loss > epoch_val_loss:
                            self.best_loss = epoch_val_loss
                            best_model_state = deepcopy(self.model.state_dict())
                            best_model_state_epoch = epoch
                    pbar.set_postfix({'Epoch Loss': epoch_val_loss})
                
                if epoch - best_model_state_epoch > self.early_stopping_wait: # Early stopping
                    break
        if fixed is None:   # 打印提前停止训练时的轮数
            print(f"Early stoping at epoch {epoch}")
            print(f"Best model at epoch {best_model_state_epoch} with loss {self.best_loss}")
            self.model.load_state_dict(best_model_state)
            # 保存模型
            torch.save(self.model.state_dict(), f'./save_model/{self.DatasetName}_best_model_sd_epoch{best_model_state_epoch}.pth')

    # 对一个有向无环图（DAG）进行拓扑排序
    def topological_ordering(self, X, step = None, eval_batch_size = None):
        
        if eval_batch_size is None:
            eval_batch_size = self.batch_size   # 表示每个批次的数据量
        eval_batch_size = min(eval_batch_size, X.shape[0])

        X = X[:self.batch_size]
        
        self.model.eval()
        order = []  # 表示拓扑序列
        parallel_order = []  # 自定义并行拓扑序列
        active_nodes = list(range(self.n_nodes))    # 表示当前未排序的节点列表
        
        
        steps_list = [step] if step is not None else range(0, self.n_steps+1, self.n_steps//self.n_votes)   # 表示用来计算雅可比矩阵的时间步列表

        if self.sorting:
            steps_list = [self.n_steps//2]
        pbar = tqdm(range(self.n_nodes-1), desc="Nodes ordered ")
        pbar = tqdm(range(self.n_nodes-1), desc="Nodes ordered ", disable=True)
        leaf = None

        print("steps_list:")
        for i, steps in enumerate(steps_list):
            print(i, steps)
        print("-------------")

        for jac_step in pbar:   # 使用一个循环来进行排序
            leaves = []
            Normal_jacob_sum = []
            for i, steps in enumerate(steps_list):  # 使用一个内部循环来遍历时间步列表
                data_loader = torch.utils.data.DataLoader(X, eval_batch_size, drop_last = True)  # 创建一个数据加载器，用来从图数据中按批次提取数据

                model_fn_functorch = self.get_model_function_with_residue(steps, active_nodes, order)   # 调用一个函数，用来获取模型在给定时间步和未排序节点上的输出函数
                leaf_ = self.compute_jacobian_and_get_leaf(data_loader, active_nodes, model_fn_functorch)   # 调用另一个函数，用来计算输出函数的雅可比矩阵，并从中选择一个叶子节点（没有后继节点的节点）
                if self.custom:
                    Normal_jacob_sum.append(leaf_)  # 这里return的leaf_实际上是归一化后的雅可比对角值

                if self.sorting:
                    order = leaf_.tolist()
                    order.reverse()
                    return order
                leaves.append(leaf_)    # 将叶子节点加入到一个列表leaves中

            if self.custom:
                Normal_jacob_sum = np.sum(Normal_jacob_sum, axis=0)
                sorted_nodes = np.argsort(Normal_jacob_sum)
                print("----------------")
                print("jacob_score", Normal_jacob_sum[sorted_nodes])
                print("local_node", sorted_nodes)
                print("----------------")

                '''
                改进思路：如果叶节点的分数相差不大就认为是并列的叶节点，应该一起用中括号括起来
                关键点：如何动态设定这个阈值呢？
                '''
                level = []
                first_leaf = sorted_nodes[0]
                # leaf_global = active_nodes[leaf]
                level.append(active_nodes[first_leaf])
                active_nodes.pop(first_leaf)

                # if len(sorted_nodes) > 1:
                #     # 增加并列的叶节点
                #     for i in range(1, len(sorted_nodes)):
                #         second_leaf = sorted_nodes[i]
                #         var = Normal_jacob_sum[second_leaf]
                #         level.append(active_nodes[second_leaf])




                parallel_order.append(level)
                if len(active_nodes) == 0:
                    break

            else:
                leaves_count = []
                for leave in leaves:
                    leaves_count.append(active_nodes[leave])
                print("出现次数global_leaves: ", leaves_count)


                leaf = Counter(leaves).most_common(1)[0][0]  # 从列表leaves中找出出现次数最多的叶子节点
                leaf_global = active_nodes[leaf]    # 将其对应的全局节点编号赋值给变量leaf_global
                order.append(leaf_global)   # 将变量leaf_global加入到拓扑序列order中
                active_nodes.pop(leaf)  # 从未排序节点列表active_nodes中删除变量leaf_global
                print("剩余节点: ", active_nodes)
                print("最终选择的leaf", leaf_global)
                print("####################################")

        if self.custom:
            if len(active_nodes) > 0:
                parallel_order.append(active_nodes)
            parallel_order.reverse()
            print("parallel_order: ", parallel_order)
            return parallel_order

        order.append(active_nodes[0])   # 将剩余的未排序节点加入到拓扑序列中，并将拓扑序列反转
        order.reverse()

        return order

    def get_model_function_with_residue(self, step, active_nodes, order):
        t_functorch = (torch.ones(1)*step).long().to(self.device) # test if other ts or random ts are better, self.n_steps
        # 定义两个匿名函数get_score_active和get_score_previous_leaves，分别用来获取模型在给定输入和时间步上对未排序节点和已排序节点的预测分数
        get_score_active = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,active_nodes]
        get_score_previous_leaves = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,order]
        def model_fn_functorch(X):
            score_active = get_score_active(X).squeeze()    # 调用匿名函数get_score_active，得到一个二维张量score_active，表示模型对未排序节点的预测分数

            if self.residue and len(order) > 0: # 如果启用了greedy并且已排序节点列表不为空
                score_previous_leaves = get_score_previous_leaves(X).squeeze()  # 调用匿名函数get_score_previous_leaves，得到一个二维张量score_previous_leaves，表示模型对已排序节点的预测分数
                jacobian_ = jacfwd(get_score_previous_leaves)(X).squeeze()  # 使用前向自动微分（jacfwd）函数，得到一个三维张量jacobian_，表示模型对已排序节点的预测分数的雅可比矩阵
                if len(order) == 1:
                    jacobian_, score_previous_leaves = jacobian_.unsqueeze(0), score_previous_leaves.unsqueeze(0)
                score_active += torch.einsum("i,ij -> j", score_previous_leaves/jacobian_[:, order].diag(), jacobian_[:, active_nodes])  #

            return score_active
        return model_fn_functorch

    def get_masked(self, x, active_nodes):
        dropout_mask = torch.zeros_like(x).to(self.device)
        dropout_mask[:, active_nodes] = 1
        return (x * dropout_mask).float()

    # 雅可比矩阵是一个向量值函数的所有一阶偏导数的矩阵
    def compute_jacobian_and_get_leaf(self, data_loader, active_nodes, model_fn_functorch):
        jacobian = []  # 初始化一个空列表jacobian，用来存储每个批次数据的雅可比矩阵
        for x_batch in data_loader:  # 遍历数据加载器data_loader中的每个批次数据x_batch
            x_batch_dropped = self.get_masked(x_batch, active_nodes) if self.masking else x_batch  # 对于每个批次数据，如果self.masking为真，就调用self.get_masked函数，用active_nodes作为掩码来过滤x_batch中的元素；否则，就直接使用x_batch。
            jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch_dropped.unsqueeze(1)).squeeze()  # 调用vmap和jacrev组合，传入model_fn_functorch作为参数，对x_batch_dropped进行向量化和反向雅可比矩阵计算。得到的结果jacobian_是一个四维张量，需要用squeeze方法去掉多余的维度。
            jacobian.append(jacobian_[..., active_nodes].detach().cpu().numpy())
        jacobian = np.concatenate(jacobian, 0)  # 用np.concatenate方法将jacobian列表中的所有数组沿着第0轴拼接起来，得到最终的雅可比矩阵jacobian。
        leaf = self.get_leaf(jacobian)  # 调用self.get_leaf函数，传入jacobian作为参数，得到叶子节点leaf，并返回。
        return leaf

    # 用来从一个雅可比矩阵中获取叶子节点。
    def get_leaf(self, jacobian_active):
        jacobian_var = jacobian_active.var(0)  # 调用了numpy库2的var方法，传入0作为axis参数，表示沿着第0轴（行）计算每一列的方差。方差是一种衡量数据分散程度的统计量。将得到的结果赋值给jacobian_var，它是一个一维数组，表示每一列的方差。
        jacobian_var_diag = jacobian_var.diagonal()
        var_sorted_nodes = np.argsort(jacobian_var_diag)
        sorted_jacob = jacobian_var_diag[var_sorted_nodes]
        if self.custom:
            np.set_printoptions(suppress=True, precision=6, linewidth=10000)  # 关闭print科学计数法
            Normal_jacob = jacobian_var_diag/np.sum(jacobian_var_diag)  # 归一化
            # print("Sorted_jacob: ", sorted_jacob)
            # print("Normal_jacob: ", Normal_jacob)
            # print("var_sorted_nodes: ", var_sorted_nodes)
            return Normal_jacob
        if self.sorting:
            return var_sorted_nodes
        leaf_current = var_sorted_nodes[0]
        return leaf_current
