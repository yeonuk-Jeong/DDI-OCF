import world
import utils
from world import cprint
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import Procedure
import pickle

from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from torch import nn

Recmodel = register.MODELS[world.model_name](world.config, dataset)
if world.parallel == True:
    Recmodel = nn.DataParallel(Recmodel, device_ids=[0, 1])
Recmodel = Recmodel.to(world.device)
if world.config['learnable_time'] == False:
    bpr = utils.BPRLoss(Recmodel, world.config)
elif world.config['learnable_time'] == True:
    bpr_t = utils.BPRLossT(Recmodel, world.config)

weight_file = utils.getFileName()

if world.config['load_embedding'] == True:
    #drug structure embedding of TWOSIDES data
    fname = f"../data/ocr-ddi/twosides/drug_embedding.pkl" 
    try:
        print("Found them! Loading embedding")
        with open(fname, "rb") as f:
            embedding = pickle.load(f)
    except FileNotFoundError:
        print(f"{fname} not exists, check your path")
        
    embedding = torch.FloatTensor(embedding).to(world.device)
    embedding = nn.functional.normalize(embedding)
    Recmodel.embedding_str = nn.Parameter(embedding)

if world.config['learnable_time'] == True:
    pretrained_weight_file = utils.getPretrainedFileName(world.config['pretrained_file_name'])
    print(f"load and save to {pretrained_weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(pretrained_weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {pretrained_weight_file}")
    except FileNotFoundError:
        print(f"{pretrained_weight_file} not exists, start from beginning")
Neg_k = 1

save_name = time.strftime("%m-%d-%Hh%Mm-") + "-" + world.dataset + "-" + world.model_name + "-" + world.config['solver'] + "-adjoint_" + str(world.adjoint) + "-learnable_t_" + str(world.config['learnable_time']) + "-dual_res_" + str(world.config['dual_res']) + "-lr" + str(world.config['lr']) + "-lr_time" + str(world.config['lr_time']) + "-decay" + str(world.config['decay'])+ "-" + world.comment

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, save_name))
else:
    w = None
    world.cprint("not enable tensorflowboard")
    
results =[]
rating_lists =[]

try:
    for epoch in range(world.TRAIN_epochs+1):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            result, rating_list = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            results.append(result)
            rating_lists.append(rating_list)
            
            

        if world.model_name == 'ltocf':
            if world.config['learnable_time'] == False:
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            else:
                output_information, times_list= Procedure.BPR_train_ode(dataset, Recmodel, bpr_t, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                print(times_list)
        else:
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        if world.config['pretrain'] == 0:
            torch.save(Recmodel.state_dict(), weight_file)
        else:
            torch.save(Recmodel.state_dict(), weight_file+'pretrained')
finally:
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('rating_lists.pkl', 'wb') as f:
        pickle.dump(rating_lists, f)
 
    if world.tensorboard:
        w.close()