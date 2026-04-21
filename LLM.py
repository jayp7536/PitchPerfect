#  WHO= Jonah
#   container for LLM class(es)
from transformers import BertTokenizer, BertModel
import torch
from dataloader import dataloaderLANG
from jeb382private import Bertize #private file



#helper module
class Squeezer(torch.nn.Module):
    def __init__(self):
        super(Squeezer, self).__init__()
    def forward(self, input):
        return input.squeeze()



#==============================================================================
#idk waht to do with this tbh, looking into another LLM
class LLM_Model():
    def __init__(self):
        pass
    
    
    
#==============================================================================
#BERT LLM model
class LLMBERT_Model():
    def __init__(self,num_classes=20, betterbert=False):
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
        if self.betterbert: SemEmb = Bertize(self.BERTmodel,self.tokenizer,rawtext, ChunkSize,overlap,extrapool)
        else:
            token_ids= self.tokenizer.tokenize(SemEmb)
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)
            if len(token_ids) >512: raise RuntimeError(f"[Text_DOMembeddings]  : text is longer than 512 tokens, <{len(token_ids)}> too large for BERT")
            word_to_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            input_ids = torch.tensor([token_ids])
            with torch.no_grad(): token_embeddings = self.model(input_ids).last_hidden_state.squeeze(0)
            
        return self.merge_head(token_embeddings)
    
    
    
#==============================================================================
import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



#helper func
def getAUC(y_true, y_score):
    try: return roc_auc_score(y_true, y_score, multi_class='ovr')
    except ValueError: return roc_auc_score(y_true, y_score[:, 1])  


def train_BERT_model(betterbert,epoch,lr,datasplit):
    data = dataloaderLANG(datasplit)
    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY
    
    
    
    #===============
    #   training
    #===============
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

        for input, labels in zip(X_train,y_train):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Accumulate loss
            running_loss += loss.item()
            
            # Store true and predicted labels for AUC computation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

        # Compute AUC
        auc_train = getAUC(np.array(y_true), np.array(y_pred))
        train_loss = running_loss / len(X_train)
        #----------
        stopwatch=time.monotonic()-stopwatch
        print(  f"Epoch [{current_epoch + 1}/{epoch}]  -  Train Loss: {train_loss:.5f}  -  Train AUC: {auc_train:.5f}  -  {(stopwatch/60):.4f}m")    #jeb382
        scheduler.step()
    
    
    
    #===============
    #   validation
    #===============
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input, labels in zip(X_val,y_val):
            outputs = model(input)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    
    
    
    #===============
    #   metrics
    #===============
    print(f"MODEL: BERT   -   better?={betterbert}")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=data.encoder.classes_))
    
    
    
    
#==============================================================================
if __name__ == "__main__":
    train_BERT_model()