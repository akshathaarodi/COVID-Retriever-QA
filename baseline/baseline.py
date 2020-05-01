#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pprint import pprint
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hashlib
import torch
from transformers import (
    BertModel,
    BertTokenizer,
)
from typing import List
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import json
import os
from tqdm.auto import tqdm


# # Load model

# In[47]:


from train_retriever_classification_new import *

encoder_question = BertEncoder(bert_question, max_question_len_global)
encoder_paragarph = BertEncoder(bert_paragraph, max_paragraph_len_global)
ret = Retriver(encoder_question, encoder_paragarph, tokenizer)

checkpoint_callback = ModelCheckpoint(
    filepath='out_v1/{ranking-epoch}-{val_loss:.2f}-{val_acc:.2f}',
    save_top_k=10,
    verbose=True,
    monitor='val_acc',
    mode='max'
)

early_stopping = EarlyStopping('val_acc', mode='max')

trainer = pl.Trainer(
    gpus=0,
#     distributed_backend='dp',
    val_check_interval=0.01,
    min_epochs=1,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=early_stopping,
    gradient_clip_val=0.5
)
ret_trainee = RetriverTrainer(ret)

tmp = torch.load('out/crossentropy-epoch=0-val_loss=0.79-val_acc=0.68.ckpt', map_location='cpu')

ret_trainee.load_state_dict(tmp['state_dict'])


# # Verify BERT Validation Performance

# In[3]:


import json


# In[4]:


def remove_html_toks(s):
    html_toks = [
        '<P>',
        '</P>',
        '<H1>',
        '</H1>',
        '</H2>',
        '</H2>',
        '<a>',
        '</a>',
        '\xa0',
        '\n'
    ]
    for i in html_toks:
        s = s.replace(i, '')
    return s


# In[1]:


hits = []
logits = []
c = 0

# # Verify Covid results

# In[6]:
with open('faq_dev.json','r') as faq:
    j = json.load(faq)


from elasticsearch import Elasticsearch


# In[7]:


es = Elasticsearch(
        [{"host": "es-covidfaq.dev.dialoguecorp.com", "port": 443}],
        use_ssl=True,
        verify_certs=True,
    )
if not es.ping():
    raise ValueError(
        "Connection failed, please start server at localhost:9200 (default)"
    )


# In[8]:


def search_section_index(es, index, query, topk):
    res = es.search(
        {
            "query": {
                "multi_match": {"query": query, "fields": ["section", "content"],}
            },
            "size": topk,
        },
        index=index,
    )
    return res


# In[14]:


secindex = "en-covid-section-index"
topk_sec = 3


# In[15]:


qs = [
    'What is Covid-19?',
    'What are the symptoms of Covid 19? ',
    'How does Covid-19 spread?',
    'When should I go to the hospital?',
    'How many cases in Montreal?',
    'What should I do if I have fever?',
    'What is the incubation period for COVID-19?',
    'How can I protect myself from the covid-19?',
    'How can I make the difference between a cold and covid19?',
    'Where can I get tested?',
    'Is it true that warm kills Coronavirus?',
]


# In[20]:
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

#rs_rerank[1]


# In[22]:

from tqdm import tqdm
c = 0 
correct = 0
for entry in tqdm(j):
    c += 1
    q = entry[0]
    answers = entry[1]
    correct_ans = answers[0]
    print('question: ', q, '\n')
    print("Expected answer is")
    print(answers[0])

    rs = search_section_index(es, secindex, q, topk_sec)["hits"]["hits"]
    
    txts = [' '.join(i['_source']['content']) + '\n'*5 for i in rs]
    
    rs_rerank = ret_trainee.retriever.predict(q, txts)
    print("first statement is")
    print(rs_rerank[0][0])
    predicted_ans = rs_rerank[0][0]
    jindex = jaccard_similarity(correct_ans.split(), predicted_ans.split())
    print("J index is",jindex)
    if jindex>0.3:
        correct += 1
        print("Answes matched")
    else:
        print("Answes mismatch")

print("Accuracy is", correct/c)


# In[ ]:




