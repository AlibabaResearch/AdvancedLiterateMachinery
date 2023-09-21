""" Copied and modified from https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/prediction.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNNAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embed_lm=False):
        super(RNNAttention, self).__init__()
        emb_dim_lm = num_classes if embed_lm else 0
        self.embed_lm = embed_lm
        self.attention_cell = AttentionCell(input_size, hidden_size, emb_dim_lm)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

        self.eos_emb = nn.Parameter(torch.ones(input_size))
        trunc_normal_(self.eos_emb, std=.02)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, mask, text=None, num_steps=26):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [b, c, h, w]
            mask: (b, h, w)
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        b, c, h, w = batch_H.size()
        mask = mask.flatten(1) # [b, N], N = h x w
        mask_pad = (1 - mask).round().bool() # inverted mask, [b, N]
        mask_pad = torch.cat([mask_pad, torch.zeros(b, 1, device=batch_H.device).bool()], dim=1) # [b, N+1]
        batch_H = batch_H.flatten(2).transpose(1, 2)
        batch_H = torch.cat([batch_H, self.eos_emb.unsqueeze(0).expand(b, -1).unsqueeze(1)], dim=1) # [b, N+1, c]

        output_hiddens = torch.FloatTensor(b, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(b, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(b, self.hidden_size).fill_(0).to(device))

        char_maps = []
        if self.training:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                if self.embed_lm:
                    char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                else:
                    char_onehots = None
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, mask_pad, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                char_maps.append(alpha)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(b).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(b, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                if self.embed_lm:
                    char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                else:
                    char_onehots = None
                hidden, alpha = self.attention_cell(hidden, batch_H, mask_pad, char_onehots)
                char_maps.append(alpha)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        char_maps = torch.cat(char_maps, dim=-1).transpose(1, 2)
        ret_dict = dict(
            logits=probs,
            char_maps=char_maps,
        )
        return ret_dict


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=0):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.num_embedding = num_embeddings # 0 denotes that LM is not used

    def forward(self, prev_hidden, batch_H, mask, char_onehots=None):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step x 1
        e.masked_fill_(mask.unsqueeze(-1), float('-inf'))

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        if self.num_embedding > 0:
            context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha
