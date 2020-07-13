import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import sys
import os
import random
import json
from queue import Queue
from math import log

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch size, sequence len, vocab size]
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # outputs 是最上層RNN的輸出
            
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        #self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        self.input_dim = emb_dim
        self.att_dim = hid_dim*2 if isatt else 0
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim + self.att_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # Decoder 只會是單向，所以 directions=1
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        '''
        first run nn
        if self.isatt:
            attn = self.attention(encoder_outputs, hidden)
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
            after = torch.cat((embedded, attn), 2)
            #print(after.shape)
        
        output, hidden = self.rnn(after, hidden)
        '''
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]
        # new put at back
        if self.isatt:
            attn = self.attention(encoder_outputs, hidden)
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
            output = torch.cat((output, attn), 2)
        
        # 將 RNN 的輸出轉為每個詞出現的機率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.dropout(output)
        output = self.embedding2vocab2(output)
        output = self.dropout(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim * 2
        self.seq_len = 50
        # for nn #
        # self.att = nn.Conv1d(self.seq_len*self.hid_dim*2, self.seq_len, 1, stride = self.seq_len, bias = False)

    
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim (* directionsssssss)]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        '''
        # for nn input layer #
        layer = decoder_hidden[0, :, :]
        concat = torch.empty(0).cuda()
        for i in range(self.seq_len):
            tmp = torch.cat((encoder_outputs[:, i, :], layer), 1)
            concat = torch.cat((concat, tmp), 1)
        #print(concat.shape)
        out = F.softmax(self.att(concat.unsqueeze(2)), dim = 1).squeeze(2)
        #print(out.shape)
        out = torch.bmm(out.unsqueeze(1), encoder_outputs)
        out = F.relu(out)
        #print(out.shape)
        '''
        # put at back dot product #
        layer = decoder_hidden[2, :, :]
        out = torch.bmm(layer.unsqueeze(1), encoder_outputs.transpose(1, 2)).squeeze(1)
        # out 60 * 50
        # out = F.softmax(out, dim = 1)
        out = torch.bmm(out.unsqueeze(1), encoder_outputs)
        return out


class beam_node():
    def __init__(self, parent, hidden, input, prob, length, output, pred):
        self.parent = parent
        self.hidden = hidden
        self.input = input
        self.prob = prob
        self.length = length
        self.output = output
        self.pred = pred

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
                "Encoder and decoder must have equal number of layers!"
            
    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds


    def inference(self, input, target, beam_size = 5):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1  
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        q = Queue()
        q.put(beam_node(None, hidden, input, 0, 1, None, []))
        preds = []
        out = []
        while not q.empty():
            candi = []
            for i in range(q.qsize()):
                obj = q.get()
                if obj.length >= input_len:
                    out.append(obj)
                    continue
                k = obj.input
                #print(k.shape, k)
                output, hidden = self.decoder(obj.input, obj.hidden, encoder_outputs)
                r = nn.Softmax(dim = 1)
                prob, top = torch.topk(r(output), beam_size)
                prob = prob.squeeze(0)
                top = top.squeeze(0)
                #print(prob.shape, top.shape)
                for j in range(beam_size):
                    #print(prob[j].item())
                    tmp_pred = torch.tensor([top[j]]).cuda()
                    tmp_pred = tmp_pred.unsqueeze(1)
                    candi.append(beam_node(obj, hidden, torch.tensor([top[j]]).cuda(), obj.prob+log(prob[j].item()), obj.length+1, output, obj.pred + [tmp_pred]))
            topk_candi = sorted(candi, key = lambda a:a.prob, reverse = True)
            for i in range(min(len(topk_candi), beam_size)):
                q.put(topk_candi[i])
        out = sorted(out, key = lambda a:a.prob, reverse=True)
        current = out[0]
        cnt = input_len-1
        while cnt > 0:
            outputs[:, cnt] = current.output
            current = current.parent
            cnt -= 1
        return outputs, torch.cat(out[0].pred, 1)

