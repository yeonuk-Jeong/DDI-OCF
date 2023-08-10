import world
import dataloader
import model
from pprint import pprint

dataset = dataloader.Loader(path="../data/"+world.dataset)

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print("adjoint method:", world.adjoint)
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    #'lgn': model.LightGCN,
    'ddiocf': model.DDIOCF,
    #'ltocf2': model.LTOCF2,
    #'ltocf1': model.LTOCF1
}