"""
Every dataset's index has to start at 0
and must be the set of consecutive natural numbers
"""
from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import networkx as nx
import pickle


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    

    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError



class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/ocr-ddi"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_drugs = 0
        train_file = path + '/drugbank_train.txt'
        test_file = path + '/drugbank_test.txt'
        self.path = path

        self.traindataSize = 0
        self.testDataSize = 0
        
        ############
        # train test 받아서 그래프 만듦
        self.train_graph = nx.read_adjlist(train_file) #그래프로 만들어서 받기 때문에 txt 파일 전처리 필요없음
        nodes = self.train_graph.nodes()
        #Add a self-loop edge to each node
        for node in nodes:
            self.train_graph.add_edge(node, node)
        
        self.test_graph = nx.read_adjlist(test_file)
        
        self.trainUniqueUsers = list(map(int, self.train_graph.nodes))
        self.testUniqueDrugs =list(map(int, self.test_graph.nodes))
        
        
        self.n_drugs = len(set(self.train_graph.nodes | self.test_graph.nodes))
        
        
        
        H = nx.Graph()
        H.add_nodes_from(self.train_graph.nodes) #노드는 train test 전체 노드
        H.add_nodes_from(self.test_graph.nodes)
        H.add_edges_from(self.train_graph.edges)
        self.DrugNet = H

        numbers = list(self.DrugNet.nodes)
        numbers.sort(key=int)
        self.adj_mat = nx.adjacency_matrix(self.DrugNet, nodelist=numbers, dtype = np.float32)
        print(self.adj_mat.shape)
        
        
        self.Graph = None
        self.getSparseGraph()
        
        
        

        self._allPos = self.getUserPosItems(list(range(self.n_drugs)))
        self.__testDict = self.__build_test()
        with open('test_set_dict.pkl', 'wb') as f:
            pickle.dump(self.__testDict, f)
        print(f"{world.dataset} is ready to go")


    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
          
        print("generating adjacency matrix")
        s = time()

        rowsum = np.array(self.adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(self.adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end-s}s, saved norm_mat...")
        sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)


        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(world.device)
        print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for drug in self.testUniqueDrugs:
            for neighbor in self.test_graph.neighbors(str(drug)):
                if test_data.get(drug):
                    test_data[drug].append(int(neighbor))
                else:
                    test_data[drug] = [int(neighbor)]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, drugs): #0 부터 순서대로 감.
        posItems = []
        for drug in drugs:
            posItems.append(self.adj_mat[drug].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
