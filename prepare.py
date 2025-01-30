import networkx as nx
import numpy as np
import random
import pickle
import os

data_dir = 'data/'
#随机流的取值范围
lb_random_flow=5
ub_random_flow=10
#随机权重的取值范围
lb_random_weigted=1
ub_random_weigted=10
MaxWeight=256
#权重矩阵
capacity_matrix = [
    [0,1000,0,390,340,0,0,410,0,0,0],
    [1000,0,400,750,0,0,760,0,0,0,0],
    [0,400,0,600,0,0,320,900,0,710,0],
    [390,750,600,0,200,310,0,0,0,0,0],
    [340,0,0,200,0,500,0,270,0,0,850],
    [0,0,0,310,500,0,730,370,440,0,0],
    [0,760,320,0,0,730,0,0,600,350,0],
    [410,0,900,0,270,370,0,0,590,0,810],
    [0,0,0,0,0,440,600,590,0,710,500],
    [0,0,710,0,0,0,350,0,710,0,720],
    [0,0,0,0,850,0,0,810,500,720,0]
]
node_num=len(capacity_matrix)

#随机生成各边权重
def random_weigted_matrix(capacity_matrix,min,max):
    weighted_edges=np.random.randint(1,10,size=(len(capacity_matrix),len(capacity_matrix)))
    for i in range(len(weighted_edges)):
        weighted_edges[i][i]=0
        for j in range(i+1,len(weighted_edges)):
            if capacity_matrix[i][j]!=0:
                weighted_edges[i][j]=random.randint(min,max)
                weighted_edges[j][i]=weighted_edges[i][j]
            else:
                weighted_edges[i][j]=MaxWeight
                weighted_edges[j][i]=weighted_edges[i][j]
    return weighted_edges

#根据权重网络G生成从小至大的k条备选路径
def k_shortest_paths(G,source,target,k):
    return list(nx.shortest_simple_paths(G, source, target, weight='weight'))[:k]

#根据权重生成G，格式为(i,j,n)表示节点i与j之间有权重为k的路径
def G_create(weighted_edges):
    G = nx.DiGraph()
    for i in range(len(weighted_edges)):
        for j in range(len(weighted_edges)):
            if 0<weighted_edges[i][j]<256:
                G.add_weighted_edges_from([(i,j,weighted_edges[i][j])])
    return G

#生成两个节点对间的随机流量
def flow_create(node_num,min,max):
    flow=[[0 for i in range(node_num)] for j in range(node_num)]
    for i in range(node_num):
        for j in range(node_num):
            flow[i][j]=random.randint(min,max)
        flow[i][i]=0
    return flow

#paths[i][j]里的列表存储i到j节点的k条备选路由
def k_paths_matrix(G,node_num,k):
    paths=[[0 for i in range(node_num)] for j in range(node_num)]
    for i in range(node_num):
        for j in range(node_num):
            if i!=j:
                paths[i][j]=(k_shortest_paths(G,i,j,k))
            else:
                paths[i][j]=[0]
    #('k-path',paths)
    return paths

#将k-paths转化为字典形式存储
#load_matrix[i][j]中存储多个字典，字典格式为{'src','dst','demand','choice'}
#表示节点i到j为原节点src到目的节点dst第choice条备选路径的途径路径，流经流量为demand
def generate_load_matrix(k_paths,random_flow):
    load_matrix=[[[] for i in range(len(k_paths))] for j in range(len(k_paths[0]))]
    for s in range(len(k_paths)):
        for p in range(len(k_paths[s])):
            if(s!=p):
                for choice in range(len(k_paths[s][p])):
                    temp=k_paths[s][p][choice]
                    for i in range(len(temp)-1):
                        load_matrix[temp[i]][temp[i+1]].append({'src':s,'dst':p,'choice':choice,'demand':random_flow[s][p]})
    return load_matrix

if __name__ == "__main__":
    k=0
    user_input=input('是否重新生成网络流和链路权重(y/n):')
    if(user_input=='y'):
        while k!=3 and k!=5 and k!=7:
            k=int(input('备选路径数量(3/5/7):'))
        weighted_edges=random_weigted_matrix(capacity_matrix,lb_random_weigted,ub_random_weigted)
        G=G_create(weighted_edges)
        k_paths=k_paths_matrix(G,node_num,k)
        random_flow=flow_create(node_num,lb_random_flow,ub_random_flow)
        load_matrix=generate_load_matrix(k_paths,random_flow)

        file_data = {
        'k_paths.pkl': k_paths,
        'random_flow.pkl': random_flow,
        'weighted.pkl': weighted_edges,
        'load_matrix.pkl':load_matrix,
        'k.pkl':k,
        'capacity.pkl':capacity_matrix,
        }
        for filename, content in file_data.items():
            with open(os.path.join(data_dir, filename),'wb') as fp:
                pickle.dump(content,fp)
                fp.close
        print('--already prepare--')
        
    elif(user_input=='n'):
        while (k!=3 and k!=5 and k!=7):
            k=int(input('备选路径数量(3/5/7):'))
        fp=open(os.path.join(data_dir, 'random_flow.pkl'), 'rb')
        random_flow=pickle.load(fp)
        fp.close
        fp=open(os.path.join(data_dir, 'weighted.pkl'), 'rb')
        weighted_edges=pickle.load(fp)
        fp.close
        fp=open(os.path.join(data_dir, 'load_matrix.pkl'), 'rb')
        load_matrix=pickle.load(fp)
        fp.close

        G=G_create(weighted_edges)
        k_paths=k_paths_matrix(G,node_num,k)
        load_matrix=generate_load_matrix(k_paths,random_flow)
        file_data = {
        'k_paths.pkl': k_paths,
        'load_matrix.pkl':load_matrix,
        'k.pkl':k
        }
        for filename, content in file_data.items():
            with open(os.path.join(data_dir, filename),'wb') as fp:
                pickle.dump(content,fp)
                fp.close
        print('--already prepared--')
    else:
        print('输入错误')