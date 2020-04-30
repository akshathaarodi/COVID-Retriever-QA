import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_reranker.models.bert_encoder import get_ffw_layers, get_cnn_layers
from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class Retriever(nn.Module):
    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(Retriever, self).__init__()
        self.bert_question_encoder = bert_question_encoder
        self.bert_paragraph_encoder = bert_paragraph_encoder
        self.tokenizer = tokenizer
        self.debug = debug
        self.max_question_len = max_question_len
        self.max_paragraph_len = max_paragraph_len
        self.cache_hash2str = {}
        self.cache_hash2array = {}

    def forward(self, **kwargs):
        raise ValueError('not implemented - use a subclass.')

    def compute_emebeddings(
            self, input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs):

        batch_size, num_document, max_len_size = batch_input_ids_paragraphs.size()

        if self.debug:
            for i in range(batch_size):
                question = self.tokenizer.convert_ids_to_tokens(
                    input_ids_question.cpu().numpy()[i])
                logger.info('>> {}'.format(question))
                for j in range(num_document):
                    answer = self.tokenizer.convert_ids_to_tokens(
                        batch_input_ids_paragraphs.cpu().numpy()[i][j])
                    logger.info('>>>> {}'.format(answer))

        h_question = self.bert_question_encoder(
            input_ids=input_ids_question, attention_mask=attention_mask_question,
            token_type_ids=token_type_ids_question)

        batch_input_ids_paragraphs_reshape = batch_input_ids_paragraphs.reshape(
            -1, max_len_size)
        batch_attention_mask_paragraphs_reshape = batch_attention_mask_paragraphs.reshape(
            -1, max_len_size)
        batch_token_type_ids_paragraphs_reshape = batch_token_type_ids_paragraphs.reshape(
            -1, max_len_size)

        h_paragraphs_batch_reshape = self.bert_paragraph_encoder(
            input_ids=batch_input_ids_paragraphs_reshape,
            attention_mask=batch_attention_mask_paragraphs_reshape,
            token_type_ids=batch_token_type_ids_paragraphs_reshape)
        #print("Before reshape shape is",h_paragraphs_batch_reshape.shape)
        #print("bat, num doc", batch_size, num_document)
        h_paragraphs_batch = h_paragraphs_batch_reshape.reshape(batch_size, num_document, 512, -1)
        #print("After reshape shape is", h_paragraphs_batch.shape)
        #h_paragraphs_batch = h_paragraphs_batch_reshape.reshape(batch_size, num_document, -1)
        return h_question, h_paragraphs_batch

    def predict(self, question_str: str, batch_paragraph_strs: List[str]):
        self.eval()
        with torch.no_grad():
            # TODO this is only a single batch

            paragraph_inputs = self.tokenizer.batch_encode_plus(
                 batch_paragraph_strs,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 max_length=self.max_paragraph_len,
                 return_tensors='pt'
             )

            tmp_device = next(self.bert_paragraph_encoder.parameters()).device
            p_inputs = {k: v.to(tmp_device).unsqueeze(0) for k, v in paragraph_inputs.items()}

            question_inputs = self.tokenizer.encode_plus(
                question_str, add_special_tokens=True, max_length=self.max_question_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            q_inputs = {k: v.to(tmp_device) for k, v in question_inputs.items()}

            q_emb, p_embs = self.forward(
                q_inputs['input_ids'], q_inputs['attention_mask'], q_inputs['token_type_ids'],
                p_inputs['input_ids'], p_inputs['attention_mask'], p_inputs['token_type_ids'],
            )

            relevance_scores = torch.sigmoid(
                torch.matmul(q_emb, p_embs.squeeze(0).T).squeeze(0)
            )

            rerank_index = torch.argsort(-relevance_scores)
            relevance_scores_numpy = relevance_scores.detach().cpu().numpy()
            rerank_index_numpy = rerank_index.detach().cpu().numpy()
            reranked_paragraphs = [batch_paragraph_strs[i] for i in rerank_index_numpy]
            reranked_relevance_scores = relevance_scores_numpy[rerank_index_numpy]
            return reranked_paragraphs, reranked_relevance_scores, rerank_index_numpy


class EmbeddingRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(EmbeddingRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = True

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                batch_token_type_ids_paragraphs):
        return self.compute_emebeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)


class FeedForwardRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
                 max_paragraph_len, debug, model_hyper_params, previous_hidden_size):
        super(FeedForwardRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = False

        check_and_log_hp(['retriever_layer_sizes'], model_hyper_params)
        ffw_layers = get_ffw_layers(
            previous_hidden_size * 2, model_hyper_params['dropout'],
            model_hyper_params['retriever_layer_sizes'] + [1], False)
        self.ffw_net = nn.Sequential(*ffw_layers)

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                batch_token_type_ids_paragraphs):
        q_emb, p_embs = self.compute_emebeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)
        _, n_paragraph, _ = p_embs.shape
        #[ batch 3 768]
        # ques [batch 1 768]
        concatenated_embs = torch.cat((q_emb.unsqueeze(1).repeat(1, n_paragraph, 1), p_embs), dim=2)
        # [batch 3 768*2]
        logits = self.ffw_net(concatenated_embs)
        print("logits shape",logits.shape)
        # [batch 3 1]
        p = logits.squeeze(dim=2)
        print("Squeeze shape", p.shape)
        return logits.squeeze(dim=2)



class CNNRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
                 max_paragraph_len, debug, model_hyper_params, previous_hidden_size):
        super(CNNRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = False

        check_and_log_hp(['retriever_layer_sizes'], model_hyper_params)
        # CNN commented
        
        # [batch 3 542 768]
        # [batch 3 new_dim 768]
        # pooling or flatten along 3rd dim
        # [batch 3 768]
        #out [batch 3]
        # softmax later
        input_channels = 768
        out_channel = 256
        kernel_size = 3
        stride = 1
        #(batch 3 542 768]
        #view change to bath * 3 542 768
        # should be along the sequence    

        #(batch token embedding)
        self.conv1 = nn.Conv1d(input_channels, out_channel, kernel_size, stride)
        self.pool = nn.MaxPool1d(kernel_size)
        self.conv2 = nn.Conv1d(256, 128 , kernel_size)
        
        self.fc1 = nn.Linear(128*59, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)




    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                batch_token_type_ids_paragraphs):
        q_emb, p_embs = self.compute_emebeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)
        _, n_paragraph, tokens, _ = p_embs.shape
        #print("Printing batch input",batch_input_ids_paragraphs)
        #print("Question embedding 0")
        #print(q_emb.shape) # bath 738

        # batch 30 738
        
        #print("Answer embedding o")
        #print(p_embs.shape) 
        # batch 3 512 738
        #print("Shape of unsqu q",q_emb.unsqueeze(1).repeat(1, n_paragraph,1,1)[0][0][0])
        # batch 1 30 738
        #p = p_embs[0]
        #print("One para dim",p[0][0])
        # para [ bath 3 512 768]
        # q [batch 30 768]
        concatenated_embs = torch.cat((q_emb.unsqueeze(1).repeat(1, n_paragraph, 1,1), p_embs), dim=2)
        #print("concatenated shape", concatenated_embs.shape)
        #[batch 3 542 768]
        x = concatenated_embs
        x = x.squeeze(0)
        #print("Reshapedx is",x.shape)
        x= x.view(-1,768,542)
        #print("Reshaped x is ", x.shape)
        x = self.conv1(x)
        #print("After 1 conv shape is",x.shape)
        x = self.pool(F.relu(x))
        #print("After first pool shape is",x.shape)
        x = self.conv2(x)
        #print("After second conv shape is",x.shape)
        x = self.pool(F.relu(x))
        #print("After secind pool size is",x.shape)
        x = x.view(-1,128*59)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print("Final x shape is",x.shape)
        #x= x.T
        #print("After transpose",x.shape)
       # x = F.sigmoid(x)
        #_,indices = x.max(0)
        #x = x.type(torch.FloatTensor)
        #print(indices)
        #new_x = torch.tensor(indices) 
        #print("type of indices is",type(x))
        return x.T





        '''
        #previous cnn stuff
        p = x.view(-1,85,542,768)
        #print("New shape of x is",x.shape)
        
        for i in range(3):
            x = p[i]
            x = x.view(-1,768,542)
            #print("changed shape is", x.shape)
            x = self.conv1(x)
            x = self.pool(F.relu(x))
            x = self.conv2(x)
            x = self.pool(F.relu(x))
            print("Intermittent x shape is", x.shape)
            x = x.view(-1, 16*134)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            #x = F.sigmoid(self.fc3(x))
            x = self.fc3(x)
            if i == 0:
                logits = x
            else:
                logits = torch.cat((logits,x),dim=1)
        
        #print("logit dim",logits.shape)
        #a = torch.argmax(logits,1)
        #print("argmax dim is",a)
        #print("arg nax is",torch.max(logits,1))
        return logits'''
