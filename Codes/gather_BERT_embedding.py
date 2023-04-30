'''
The following code will help extract the BERT embedding from the text
'''

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F

start = time.time()

# Change the path accordingly
path = "/media2/sadat/Sadat/EWC_coursework/Data/"

all_files = os.listdir(path)
batch_size=8 # change it according to your  
device = "cuda:0"

def create_dataloader(features, attention_masks):
    # Create the DataLoader for our training set
    '''
    This function will create a dataloader for our training set. The dataloader will help to feed the randomly 
    sampled data on each batch. The batch size is selected to be 8
    '''
    data = TensorDataset(features, attention_masks)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def prepare_BERT_input(df):
    """BERT input requires to be toekenized, and it needs attention masks as well
    (1 where there is a value, 0 where to pass)
    """
    tokenized = np.array(list(df["tokenized"]))
    attention_masks = np.where(tokenized>0, 1, 0)
    tokenized, attention_masks = torch.tensor(tokenized), torch.tensor(attention_masks)
    return tokenized, attention_masks

model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)


for idx, f in enumerate(all_files):
    # collect embeddings for all files
    if f.endswith(".pkl"):
        df = pd.read_pickle(path + f)
        tok, att = prepare_BERT_input(df)
        Loader = create_dataloader(tok, att)
        filename = f.split(".")[0]
        np_array_holder_batchwise = []
        for L in tqdm(Loader):
            feat, att = L
            with torch.no_grad():
                bert_cls_outputs = model(input_ids=feat.to(device),
                            attention_mask=att.to(device))[0][:, 0, :]
                
                np_array = bert_cls_outputs.cpu().numpy()
                np_array_holder_batchwise.append(np_array)
        full_np_array = np.concatenate(np_array_holder_batchwise, axis=0)
        #print(full_np_array.shape)
        np.savez(path + "BERT_Embeddings/" + filename, feat=full_np_array, label=df["is_deception"].values)
        print("Done with {}. Still left {} ".format(filename, len(all_files)-idx))
                





