#  WHO= Jonah
#   container for LLM class(es)
from transformers import BertTokenizer, BertModel
import torch
from dataloader import dataloaderLANG
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from jeb382private import Bertize #private file




#helper func
def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # Probabilities of the true classes
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()



#helper module
class Squeezer(torch.nn.Module):
    def __init__(self):
        super(Squeezer, self).__init__()
    def forward(self, input):
        return input.squeeze()



#============================================================================================================================================================
#idk waht to do with this tbh, looking into another LLM
#TODO: (jonah) do end of hw3 but here
class LLM_Model():
    def __init__(self):
        pass
    
    
    
#==============================================================================
def train_LLM_model(epoch=1,lr=0.01,datasplit=0.7):
    pass












#============================================================================================================================================================
#BERT LLM model
class LLMBERT_Model(torch.nn.Module):
    def __init__(self,num_classes=20, betterbert=False):
        super().__init__()
        self.betterbert=betterbert
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        self.BERTmodel = BertModel.from_pretrained("bert-large-uncased")
        
        if self.betterbert:
            self.merge_head = torch.nn.Sequential(
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024,128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
            
                torch.nn.LayerNorm(128), 
                torch.nn.Linear(128, num_classes),
                torch.nn.Sigmoid()  )
        else:
            self.merge_head = torch.nn.Sequential(
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024,1),
                Squeezer(),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                
                torch.nn.LayerNorm(512), 
                torch.nn.Linear(512, num_classes),
                torch.nn.Sigmoid()  )
    #==============================================================================
    def forward(self,rawtext,ChunkSize=512,overlap=448,extrapool=True):
        if self.betterbert: token_embeddings = Bertize(self.BERTmodel,self.tokenizer,rawtext, ChunkSize,overlap,extrapool)
        else:
            token_ids= self.tokenizer.tokenize(rawtext)
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)
            if len(token_ids) >512:
                # raise RuntimeError(f"[LLMBERT_Model]  : text is longer than 512 tokens, <{len(token_ids)}> too large for BERT")
                # print(f"WARNING   [LLMBERT_Model]  : text is longer than 512 tokens, <{len(token_ids)}> too large for BERT, cutting down to 512")
                input_ids=torch.tensor([token_ids[:512]])
            else:
                #pad!!!
                input_ids = F.pad(torch.tensor([token_ids]), (0, max(0, 512-len(token_ids))), mode='constant', value=0)
            # word_to_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            with torch.no_grad(): token_embeddings = self.BERTmodel(input_ids).last_hidden_state.squeeze(0)
            
        return self.merge_head(token_embeddings)
    
    
    
#==============================================================================
def train_BERT_model(betterbert=False,epoch=1,lr=0.01,datasplit=0.7):
    print('load data')
    data = dataloaderLANG(datasplit)
    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY
    
    
    
    #===============        Note batchsize=1s
    #   training
    #===============
    print('load model')
    model = LLMBERT_Model(betterbert=betterbert)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = lambda outputs, targets: focal_loss(outputs, targets, alpha=0.25, gamma=2.0)# 0.05 8.0 gave best at 0.83 AUC
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) #was 10 and 2
    
    print('epoching')
    for current_epoch in range(epoch):
        stopwatch = time.monotonic()    #jeb382
        #----------
        model.train()
        running_loss = 0.0
        y_true = []
        y_pred = []

        for idx,(input, labels) in enumerate(zip(X_train[:5],y_train[:5])):
            # print(f'epoch= {current_epoch}   {idx}/50')
            optimizer.zero_grad()
            
            outputs = model(input)
            labels = torch.tensor(labels).to(torch.long)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            y_true.append(labels.item())
            y_pred.append( torch.argmax(outputs, dim=0) )

        acc_train = accuracy_score(  np.array(y_true), np.array(y_pred)  )
        train_loss = running_loss / len(X_train)
        #----------
        stopwatch=time.monotonic()-stopwatch
        print(  f"Epoch [{current_epoch + 1}/{epoch}]  -  Train Loss: {train_loss:.5f}  -  Train ACC: {acc_train:.5f}  -  {(stopwatch/60):.4f}m")    #jeb382
        scheduler.step()
    
    
    
    #===============
    #   validation
    #===============
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input, labels in zip(X_val[:5],y_val[:5]):
            outputs = model(input)
            y_true.append(labels)
            y_pred.append( torch.argmax(outputs, dim=0) )
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    
    
    
    #===============
    #   metrics
    #===============
    print(f"MODEL: BERT   -   better?={betterbert}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(  classification_report(y_true, y_pred, labels=np.arange(20), target_names=data.encoder.classes_)  )












#============================================================================================================================================================
if __name__ == "__main__":
    train_BERT_model()