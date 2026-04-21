# WHO= everyone?
#   main script
from metrics import *
from ngram import train_ngram_model
from tfidf import train_tfidf_model
from LLM import train_BERT_model
from combined import train_combined_model



#change manually
MODELTYPE=0
# N=2
EPOCHS=10
# learningrate=0.1
TRAINTESTSPLIT=0.7
IFJONAH=False




if __name__ == "__main__":
    if MODELTYPE==-1:
        train_BERT_model(betterbert=IFJONAH, epoch=EPOCHS,lr=0.01,    datasplit=TRAINTESTSPLIT) #LLM-BERT
        train_BERT_model(betterbert=IFJONAH, epoch=EPOCHS,lr=0.001,   datasplit=TRAINTESTSPLIT) #LLM-BERT
        train_BERT_model(betterbert=IFJONAH, epoch=EPOCHS,lr=0.0001,  datasplit=TRAINTESTSPLIT) #LLM-BERT
        train_BERT_model(betterbert=IFJONAH, epoch=EPOCHS,lr=0.00001, datasplit=TRAINTESTSPLIT) #LLM-BERT
        train_BERT_model(betterbert=IFJONAH, epoch=EPOCHS,lr=0.000001,datasplit=TRAINTESTSPLIT) #LLM-BERT
    
    elif MODELTYPE== 0:
        train_tfidf_model   (n=1, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_tfidf_model   (n=2, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_tfidf_model   (n=3, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_tfidf_model   (n=4, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_tfidf_model   (n=5, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
    
    elif MODELTYPE== 1:
        train_ngram_model(n=1, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_ngram_model(n=2, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_ngram_model(n=3, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_ngram_model(n=4, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_ngram_model(n=5, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
    
    elif MODELTYPE== 2:
        train_combined_model(n=1, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_combined_model(n=2, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_combined_model(n=3, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_combined_model(n=4, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)
        train_combined_model(n=5, epoch=EPOCHS,datasplit=TRAINTESTSPLIT)