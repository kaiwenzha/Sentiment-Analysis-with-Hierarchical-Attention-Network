from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init
from constants import *

class HA_NET(torch.nn.Module): #Hierarchikay Attention Network
    def __init__(self, embedding_length):
        super(HA_NET, self).__init__()

        self.conv1 = nn.Conv2d(1, 100, (1, GRU_Word_Hidden_Size), stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(1, 100, (2, GRU_Word_Hidden_Size), stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(1, 100, (3, GRU_Word_Hidden_Size), stride = 1, padding = (1,0))
        self.conv4 = nn.Conv2d(1, 100, (4, GRU_Word_Hidden_Size), stride = 1, padding = (1,0))
        self.conv5 = nn.Conv2d(1, 100, (5, GRU_Word_Hidden_Size), stride = 1, padding = (2,0))
        self.conv6 = nn.Conv2d(1, 100, (6, GRU_Word_Hidden_Size), stride = 1, padding = (2,0))

        self.gru_word = nn.GRUCell(embedding_length, GRU_Word_Hidden_Size)
        self.gru_sentence = nn.GRUCell(600, GRU_Sentence_Hidden_Size)

        self.fc1 = nn.Linear(GRU_Sentence_Hidden_Size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.apply(weights_init)

        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)

        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs_all):
        # inputs_all: (num_sentence, sentence_length, dim) Tensor

        num_sentece = inputs_all.size()[0]
        sentences = []
        for idx_sentence in range(num_sentece): # generate sentence level representation
            inputs = inputs_all[idx_sentence]
            if not inputs.is_cuda:
                inputs = inputs.cuda()
            h_forward = Variable(torch.zeros(1, GRU_Word_Hidden_Size).cuda())
            h_back = Variable(torch.zeros(1, GRU_Word_Hidden_Size).cuda())

            sentence_length = inputs.size()[0]
            h_states_forward = Variable(torch.zeros(sentence_length, 1, GRU_Word_Hidden_Size).cuda())
            h_states_back = Variable(torch.zeros(sentence_length, 1, GRU_Word_Hidden_Size).cuda())
            for i in range(sentence_length):
                h_forward = self.gru_word(inputs[i].unsqueeze(0), h_forward)
                h_back = self.gru_word(inputs[sentence_length - 1 - i].unsqueeze(0), h_back)
                h_states_forward[i] = h_forward
                h_states_back[sentence_length - 1 - i] = h_back

            h_states_forward = h_states_forward.transpose(0,1) #(sentence_length, batch, dim)->(batch, sentence_length, dim)
            h_states_back = h_states_back.transpose(0,1)

            h_states = torch.cat((h_states_forward, h_states_back), 1) # ->(batch, 2*sentence_length, dim)
            h_states = h_states.unsqueeze(1) # ->(batch, C, 2*sentence_length, dim)

            sentence = torch.cat((
                torch.sigmoid(F.max_pool1d(self.conv1(h_states).squeeze(3), kernel_size=2*sentence_length)).squeeze(2),
                torch.sigmoid(F.max_pool1d(self.conv2(h_states).squeeze(3), kernel_size=2*sentence_length)).squeeze(2),
                torch.sigmoid(F.max_pool1d(self.conv3(h_states).squeeze(3), kernel_size=2*sentence_length)).squeeze(2),
                torch.sigmoid(F.max_pool1d(self.conv4(h_states).squeeze(3), kernel_size=2*sentence_length)).squeeze(2),
                torch.sigmoid(F.max_pool1d(self.conv5(h_states).squeeze(3), kernel_size=2*sentence_length)).squeeze(2),
                torch.sigmoid(F.max_pool1d(self.conv6(h_states).squeeze(3), kernel_size=2*sentence_length)).squeeze(2)),
                1) # (batch, 6*100)
            sentences.append(sentence)

        h_forward = Variable(torch.zeros(1, GRU_Sentence_Hidden_Size).cuda())
        h_back = Variable(torch.zeros(1, GRU_Sentence_Hidden_Size).cuda())
        h_sentence_forward = Variable(torch.zeros(num_sentece, 1, GRU_Sentence_Hidden_Size).cuda())
        h_sentence_back = Variable(torch.zeros(num_sentece, 1, GRU_Sentence_Hidden_Size).cuda())
        for i in range(num_sentece): #generate document level representation
            h_forward = self.gru_sentence(sentences[i], h_forward)
            h_back = self.gru_sentence(sentences[num_sentece-1-i], h_back)
            h_sentence_forward[i] = h_forward
            h_sentence_back[num_sentece - 1 - i] = h_back

        h_sentence_forward = h_sentence_forward.transpose(0,1)  # (num_sentence, batch, dim)->(batch, num_sentence, dim)
        h_sentence_back = h_sentence_back.transpose(0, 1)
        h_document = torch.cat((h_sentence_forward, h_sentence_back), 1)
        h_document = h_document.mean(1)

        x = torch.sigmoid(self.fc1(h_document))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

if __name__ == '__main__':
    c = HA_NET(256).cuda()
    print(c.forward(Variable(torch.ones(5, 30, 256)).cuda()))
