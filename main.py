# WHO= everyone?
#   main script
#       insert:
#           - what model (tf-idf, ngram with what as n)
#           - epochs
#           - learning rate
from metrics import *
from dataloader import dataloaderLANG
from ngram import ngramModel
from tfidf import tfidfModel
from LLM import LLM_Model,LLMBERT_Model


modeltype=0
epochs=10
learningrate=0.1
trainingtestingsplit=0.7



#   load model
if   modeltype==-2: model= LLM_Model() #LLM
elif modeltype==-1: model= LLMBERT_Model() #LLM-BERT
elif modeltype== 0: model= tfidfModel() #tf-idf
else:               model= ngramModel(modeltype) #n-gram, pass modeltype as n


#   load dataloader
dataloader= dataloaderLANG(trainingtestingsplit)


#   training cycle of model (training split)
for TrainX,TrainY in zip(dataloader.TrainX,dataloader.TrainY):
    pass


#   run model on testsing split
for TestX,TestY in zip(dataloader.TrainX,dataloader.TrainY):
    pass


#   get metrics
print('a')