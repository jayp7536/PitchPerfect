#  WHO= Jonah
#   container for LLM class(es)
from transformers import BertTokenizer, BertModel
import torch
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
def train_BERT_model(betterbert,epoch,lr,trainingtestingsplit):
    pass
    
    
    
    
#==============================================================================
if __name__ == "__main__":
    train_BERT_model()