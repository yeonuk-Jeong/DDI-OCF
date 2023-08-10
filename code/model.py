"""
Define models here
"""
import world
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import odeblock as ode

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getDrugsRating(self, users):
        raise NotImplementedError
        
        
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        
        self.num_drugs  = dataset.n_drugs
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_drugs = torch.nn.Embedding(
            num_embeddings=self.num_drugs, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getDrugsRating(self, drugs):
        drugs = drugs.long()
        drugs_emb = self.embedding_drugs(drugs)
        all_drugs_emb = self.embedding_drugs.weight
        scores = torch.matmul(drugs_emb, all_drugs_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, drugs, pos, neg):
        drugs_emb = self.embedding_drugs(drugs.long())
        pos_emb   = self.embedding_drugs(pos.long())
        neg_emb   = self.embedding_drugs(neg.long())
        pos_scores= torch.sum(drugs_emb*pos_emb, dim=1)
        neg_scores= torch.sum(drugs_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(drugs_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(drugs))
        return loss, reg_loss
        
    def forward(self, drugs):
        drugs = drugs.long()
        drugs_emb = self.embedding_drug(drugs)
        inner_pro = torch.mul(drugs_emb, drugs_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        
        return self.f(gamma)


class DDIOCF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(DDIOCF, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.__init_ode()


    def __init_weight(self):
        self.num_drugs  = self.dataset.n_drugs
        self.latent_dim = self.config['latent_dim_rec']
        
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_drugs = torch.nn.Embedding(
            num_embeddings=self.num_drugs, embedding_dim=self.latent_dim)
        self.embedding_str = None
        if self.config['pretrain'] == 0:

            nn.init.normal_(self.embedding_drugs.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
    
    def __init_ode(self):
        self.time_split = self.config['time_split'] # init the number of time split
        if world.config['learnable_time'] == True:
            
            self.odetimes = ode.ODETimeSetter(self.time_split, self.config['K'])
            self.odetime_1 = [self.odetimes[0]]
            self.odetime_2 = [self.odetimes[1]]
            self.odetime_3 = [self.odetimes[2]]
            self.ode_block_test_1 = ode.ODEBlockTimeFirst(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
            self.ode_block_test_2 = ode.ODEBlockTimeMiddle(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
            self.ode_block_test_3 = ode.ODEBlockTimeMiddle(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
            self.ode_block_test_4 = ode.ODEBlockTimeLast(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
        else:
            self.odetime_splitted = ode.ODETimeSplitter(self.time_split, self.config['K'])
            self.ode_block_1 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], 0, self.odetime_splitted[0])
            self.ode_block_2 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], self.odetime_splitted[0], self.odetime_splitted[1])
            self.ode_block_3 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], self.odetime_splitted[1], self.odetime_splitted[2])
            self.ode_block_4 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], self.odetime_splitted[2], self.config['K'])

    def get_time(self):
        ode_times=list(self.odetime_1)+ list(self.odetime_2)+ list(self.odetime_3)
        return ode_times
        
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods
        """       
        drugs_emb = self.embedding_drugs.weight
        if self.config['load_embedding'] == True:
            pre_emb = self.embedding_str
            drugs_emb = torch.cat([drugs_emb, pre_emb],dim=1)
        all_emb = drugs_emb
        embs = [drugs_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        """
        layers
        """
        if world.config['learnable_time'] == True:
            out_1 = self.ode_block_test_1(all_emb, self.odetime_1)
            if world.config['dual_res'] == False:
                out_1 = out_1 - all_emb
            embs.append(out_1)

            out_2 = self.ode_block_test_2(out_1, self.odetime_1, self.odetime_2)
            if world.config['dual_res'] == False:
                out_2 = out_2 - out_1
            embs.append(out_2)

            out_3 = self.ode_block_test_3(out_2, self.odetime_2, self.odetime_3)
            if world.config['dual_res'] == False:
                out_3 = out_3 - out_2
            embs.append(out_3)            

            out_4 = self.ode_block_test_4(out_3, self.odetime_3)
            if world.config['dual_res'] == False:
                out_4 = out_4 - out_3
            embs.append(out_4)
            
        elif world.config['learnable_time'] == False:
            all_emb_1 = self.ode_block_1(all_emb)
            all_emb_1 = all_emb_1 - all_emb
            embs.append(all_emb_1)
            all_emb_2 = self.ode_block_2(all_emb_1)
            all_emb_2 = all_emb_2 - all_emb_1
            embs.append(all_emb_2)
            all_emb_3 = self.ode_block_3(all_emb_2)
            all_emb_3 = all_emb_3 - all_emb_2
            embs.append(all_emb_3)
            all_emb_4 = self.ode_block_4(all_emb_3)
            all_emb_4 = all_emb_4 - all_emb_3
            embs.append(all_emb_4)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)        

        #users, items = torch.split(light_out, [self.num_users, self.num_items])
        drugs= light_out
        return drugs
    
    def getDrugsRating(self, drugs):
        all_drugs = self.computer()
        drugs_emb = all_drugs[drugs.long()]
        rating = self.f(torch.matmul(drugs_emb, all_drugs.t()))
        return rating
    
    def getEmbedding(self, drugs, pos_items, neg_items):
        all_drugs = self.computer()
        drugs_emb = all_drugs[drugs]
        pos_emb = all_drugs[pos_items]
        neg_emb = all_drugs[neg_items]
        drugs_emb_ego = self.embedding_drugs(drugs)
        pos_emb_ego = self.embedding_drugs(pos_items)
        neg_emb_ego = self.embedding_drugs(neg_items)
        return drugs_emb, pos_emb, neg_emb, drugs_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, drugs, pos, neg):
        (drugs_emb, pos_emb, neg_emb, 
        drugEmb0,  posEmb0, negEmb0) = self.getEmbedding(drugs.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(drugEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(drugs))
        pos_scores = torch.mul(drugs_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(drugs_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, drugs):
        # compute embedding
        all_drugs = self.computer()
        drugs_emb = all_drugs[drugs]

        inner_pro = torch.mul(drugs_emb, drugs_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
