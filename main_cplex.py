import networkx as nx
import time
from docplex.mp.model import Model
import pickle
import os
import matplotlib.pyplot as plt
import prepare
import numpy as np

data_dir = 'data/'
node_names = {0: 'London', 1: 'Copenhager', 2: 'Berlin',3:'Amsterdam',4:'Brussels',5:'Luxemburg',6:'Prague',7:'Paris',8:'Zurich',9:'Vienna',10:'Milan'}

#绘制网络拓扑
def draw_graph(G, edge_labels=None,save_folder=None, file_name='my_graph', file_format='png',label=None):
    plt.figure(figsize=(15,15))
    node_sizes = [len(str(label)) * 10 for node in G.nodes()]
    nx.draw(G, labels=label,node_size=node_sizes,pos=nx.circular_layout(G), with_labels=True,font_size=12,arrows=False)
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos=nx.circular_layout(G),
            font_size=10,
            edge_labels=edge_labels
        )
    if save_folder:
        file_path = os.path.join(save_folder, f"{file_name}.{file_format}")
        plt.savefig(file_path, format=file_format)
        print(f"Graph saved at {file_path}")

def cal_load(capacity_matrix,weighted_edges,A_matrix,B_matrix,b_matrix):
    mdl = Model(name="load_optimization")
    num_nodes = len(capacity_matrix)
    h_var = mdl.continuous_var(name="h")
    p_var=mdl.continuous_var_list(num_nodes*num_nodes*k,lb=0,name='p')
    capacity_upper=np.triu(capacity_matrix)
    upper_list=capacity_upper[capacity_upper!=0].tolist()

    for i in range(len(upper_list)):
        mdl.add_constraint(mdl.sum(A_matrix[i][j]*p_var[j] for j in range(len(p_var)))<=upper_list[i])
        mdl.add_constraint(mdl.sum(A_matrix[i][j]*p_var[j] for j in range(len(p_var)))/upper_list[i]<=h_var)
    for i in range(len(B_matrix)):
        mdl.add_constraint(mdl.sum(B_matrix[i][s]*p_var[s] for s in range(len(p_var)))==b_matrix[i])
    mdl.minimize(h_var)
    mdl.export_as_lp(data_dir,'original_problem')

    start_time=time.time()
    solution = mdl.solve()
    end_time=time.time()
    solve_time=end_time-start_time
    print(f"原始模型求解时间为{solve_time*1000:.4f}ms")

    if solution:
        print(f"最小化最大值:{solution.objective_value*100:.2f}%")

        xvar_matrix=[]
        for i in range(len(p_var)):
            xvar_matrix.append(solution.get_var_value(p_var[i]))
        m=0
        load_rate=[[0 for _ in range(num_nodes)]for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i+1,num_nodes):
                if(capacity_matrix[i][j]!=0):
                    load_rate[i][j]=sum(xvar_matrix[s]*A_matrix[m][s]for s in range(len(A_matrix[m])))/upper_list[m]
                    m+=1     
        G=prepare.G_create(weighted_edges)
        edge_labels = {}
        for edge in G.edges:
            i, j = edge
            if i > j: i, j = j, i
            edge_labels[edge] = '{:.1f}%'.format(load_rate[i][j]*100)
        draw_graph(G, edge_labels,data_dir,'cplex_optimized_loadrate','png',node_names)
    else:
        print("未找到可行解")
        return False  
    return xvar_matrix,load_rate

def dual_model(capacity_matrix,A_matrix,B_matrix,b_matrix):  
    mdl = Model(name="dual_optimization")
    num_nodes = len(capacity_matrix)
    capacity_upper=np.triu(capacity_matrix)
    upper_list=capacity_upper[capacity_upper!=0].tolist()
    p1_var=mdl.continuous_var_list(len(upper_list),ub=0,name='p1')
    p2_var=mdl.continuous_var_list(len(upper_list),lb=0,name='p2')
    p3_var=mdl.continuous_var_list(num_nodes*num_nodes,lb=0,name='p3')

    mdl.add_constraint((mdl.sum(p2_var[s]*upper_list[s] for s in range(len(upper_list))))<=1)
    for n in range(num_nodes*num_nodes*k):
        mdl.add_constraint(mdl.sum((p2_var[s]-p1_var[s])*A_matrix[s][n] for s in range(len(A_matrix)))+mdl.sum(-1*p3_var[s]*B_matrix[s][n] for s in range(len(B_matrix)))>=0)
    mdl.maximize(mdl.sum(p1_var[s]*upper_list[s] for s in range(len(upper_list)))+mdl.sum(p3_var[s]*b_matrix[s] for s in range(len(b_matrix))))

    mdl.export_as_lp(data_dir,'dual_problem')
    start_time=time.time()
    solution = mdl.solve()
    end_time=time.time()
    solve_time=end_time-start_time
    print(f"对偶模型求解时间为{solve_time*1000:.4f}ms")
    if solution:
        print(f"对偶模型最优值:{mdl.objective_value*100:.2f}%")
    else:
        print("未找到可行解")
        return False  

def create_problem_matrix(load_matrix,capacity_matrix):
    num_nodes = len(capacity_matrix)
    capacity_upper=np.triu(capacity_matrix)
    upper_list=capacity_upper[capacity_upper!=0].tolist()
    A_matrix=[[0 for _ in range(num_nodes*num_nodes*k)]for _ in range(len(upper_list))]
    B_matrix=[[0 for _ in range(num_nodes*num_nodes*k)]for _ in range(num_nodes*num_nodes)]
    b_matrix=[0 for _ in range(num_nodes*num_nodes)]
    s=0
    m=0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if(i!=j):
                for c in range(k):
                    B_matrix[m][(i*num_nodes+j)*k+c]=1
                b_matrix[m]=1       
            else:
                for c in range(k):
                    B_matrix[m][(i*num_nodes+j)*k+c]=1
                b_matrix[m]=0
            m+=1
            if(j>i and capacity_matrix[i][j]!=0):
                temp1 = load_matrix[i][j]
                temp2=load_matrix[j][i]
                for p in range(len(temp1)):
                    src,dst,choice,demand=temp1[p].values()
                    A_matrix[s][(src*num_nodes+dst)*k+choice]=demand
                for p in range(len(temp2)):
                    src,dst,choice,demand=temp2[p].values()
                    A_matrix[s][(src*num_nodes+dst)*k+choice]=demand
                upper_list[s]=capacity_matrix[i][j]
                s+=1
    return A_matrix,B_matrix,b_matrix

if __name__ == "__main__":
    #数据加载
    fp=open(os.path.join(data_dir, 'k_paths.pkl'), 'rb')
    k_paths=pickle.load(fp)
    fp.close
    fp=open(os.path.join(data_dir, 'random_flow.pkl'), 'rb')
    random_flow=pickle.load(fp)
    fp.close
    fp=open(os.path.join(data_dir, 'weighted.pkl'), 'rb')
    weighted_edges=pickle.load(fp)
    fp.close
    fp=open(os.path.join(data_dir, 'load_matrix.pkl'), 'rb')
    load_matrix=pickle.load(fp)
    fp.close
    fp=open(os.path.join(data_dir, 'k.pkl'), 'rb')
    k=pickle.load(fp)
    fp.close
    fp=open(os.path.join(data_dir, 'capacity.pkl'), 'rb')
    capacity_matrix=pickle.load(fp)
    fp.close

    G=prepare.G_create(weighted_edges)
    A_matrix,B_matrix,b_matrix=create_problem_matrix(load_matrix,capacity_matrix)
    print(f'备选路径数量为{k}时')
    result=cal_load(capacity_matrix,weighted_edges,A_matrix,B_matrix,b_matrix)
    if(result is not False):
        xvar_matrix,load_rate=result
        dual_model(capacity_matrix,A_matrix,B_matrix,b_matrix)