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
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score
import torch





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
class LLM_Model(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")

        # simple classifier
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,     
            padding="max_length",
            max_length= 128
        )

        outputs = self.model(**inputs)

        # take CLS token
        pooled = outputs.last_hidden_state.mean(dim=1)

        logits = self.classifier(pooled)

        return logits.squeeze(0)
    
    
#==============================================================================
def train_LLM_model(epoch=1,lr=1e-4,datasplit=0.7):
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
def train_LLM_model(epoch=1, lr=1e-4, datasplit=0.7):
    print("load data")
    data = dataloaderLANG(datasplit)

    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY

    print("load model")
    model = LLM_Model(num_classes=len(data.encoder.classes_))

    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("training...")

    model.train()
    total_loss = 0

    for text, label in zip(X_train[:200], y_train[:200]):
        optimizer.zero_grad()

        outputs = model(text)

        loss = criterion(
            outputs.unsqueeze(0),
            torch.tensor([label])
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch 1 Loss: {total_loss:.4f}")

    # ===== validation =====
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for text, label in zip(X_val[:100], y_val[:100]):
            outputs = model(text)

            y_true.append(label)
            y_pred.append(torch.argmax(outputs).item())

    print("\n#----------- LLM RESULTS -----------#")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("#----------------------------------#")












#============================================================================================================================================================
if __name__ == "__main__":
    train_LLM_model()