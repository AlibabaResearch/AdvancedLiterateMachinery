import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.nn import init
from torch.nn.parameter import Parameter

from modules.recognizer.crnn_backbone_line import FCNEncoder


class CTC(nn.Module):
    def __init__(self):
        super(CTC, self).__init__()
        self.criterion = nn.CTCLoss(reduction="none", zero_infinity=True)

    def forward(self, input, label):
        batch_size, total_len = label.size()
        label_len = np.zeros(batch_size)
        label_seq = []
        label_total = 0
        for bn in range(batch_size):
            for tn in range(total_len):
                if label[bn][tn] != 0:
                    label_len[bn] = label_len[bn] + 1
                    label_total += 1
                    label_seq.append(int(label[bn][tn]))
        label_seq = np.array(label_seq)
        label_len = Variable(
            torch.from_numpy(label_len).type(torch.IntTensor), requires_grad=False
        )
        label = Variable(
            torch.from_numpy(label_seq).type(torch.IntTensor), requires_grad=False
        )
        probs_sizes = Variable(
            torch.IntTensor([input.size(0)] * batch_size), requires_grad=False
        )
        loss = self.criterion(
            input.log_softmax(2), label, probs_sizes, label_len
        ).mean() / input.size(0)
        return loss


class ExCTC(nn.Module):
    def __init__(self, dict_file):
        super(ExCTC, self).__init__()
        nClass = len(open(dict_file, "r").readlines()) + 2  # blank and outer
        self.F = FCNEncoder()
        self.C = nn.Linear(512, nClass, bias=False)
        self.emb = nn.Embedding(nClass, 512)
        self.emb.weight = self.C.weight
        self.CTC = CTC()
        self.max_text_len = 128
        self.abs_pos_emb = nn.Embedding(self.max_text_len, 512)

    def forward(self, input, label):
        b, n, device = *label.shape, label.device
        features = self.F(input)
        features_1D = features.squeeze(2).permute(0, 2, 1)
        features_1D = F.dropout(features_1D, 0.5, training=self.training)
        output = self.C(features_1D)
        ctc_loss = self.CTC(output.permute(1, 0, 2), label)
        return {"output": output, "loss": ctc_loss}

    @torch.no_grad()
    def get_text_embed(self, text):
        b, n, device = *text.shape, text.device
        text = text[..., : self.max_text_len]
        text_mask = text != 0
        text_encodings = self.emb(text)
        return text_encodings, text_mask

    @torch.no_grad()
    def get_image_embed(self, image):
        image_embed = self.F(image)
        return image_embed

    @torch.no_grad()
    def get_image_features(self, image):
        features_1D = self.F(image)
        image_features = features_1D.squeeze(2).permute(0, 2, 1)
        return image_features
