import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from typing import Optional
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint

from .block import MLP
from tqdm import tqdm

class Transformer(nn.Module):

    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, 
                 dropout, normalize_before, pad_token_id, num_classes, max_position_embeddings, 
                 return_intermediate_dec, num_bins, pt_eos_index, activation="relu", args=None):
        super(Transformer, self).__init__()
        self.embedding = DecoderEmbeddings(num_classes, d_model, pad_token_id, max_position_embeddings, dropout)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self.pt_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)

        self.poly_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)

        self.rec_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)

        self.pt_pred_layer = MLP(d_model, d_model, num_classes, 3)
        self.poly_pred_layer = MLP(d_model, d_model, num_classes, 3)
        self.rec_pred_layer = MLP(d_model, d_model, num_classes, 3)

        self._reset_parameters()

        self.nhead = nhead
        self.d_model = d_model
        self.num_bins = num_bins
        self.pt_eos_index = pt_eos_index
        self.max_position_embeddings = max_position_embeddings

        self.args = args

        self.classes_cord = ['menu.cnt', 'menu.discountprice', 'menu.etc', 'menu.itemsubtotal', 'menu.nm', 'menu.num',
                        'menu.price', 'menu.sub.cnt', 'menu.sub.nm', 'menu.sub.price', 'menu.sub.unitprice', 'menu.unitprice', 
                        'menu.vatyn', 'sub_total.discount_price', 'sub_total.etc', 'sub_total.othersvc_price', 'sub_total.service_price',
                        'sub_total.subtotal_price', 'sub_total.tax_price', 'total.cashprice', 'total.changeprice', 'total.creditcardprice', 
                        'total.emoneyprice', 'total.menuqty_cnt', 'total.menutype_cnt', 'total.total_etc', 'total.total_price', 'void_menu.nm', 'void_menu.price']
        self.index2class_cord = {}
        for i in range(len(self.classes_cord)):
            self.index2class_cord[args.padding_index + 1 + i] = self.classes_cord[i]

        self.classes_sroie = ['company', 'address', 'date', 'total']        
        self.index2class_sroie = {}
        for i in range(len(self.classes_sroie)):
            self.index2class_sroie[args.padding_index + 1 + i] = self.classes_sroie[i]

        if args.val_dataset:
            if 'cord' in args.val_dataset[0]:
                self.index2class = self.index2class_cord
            elif 'sroie' in args.val_dataset[0]:
                self.index2class = self.index2class_sroie

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def decode(self, input_seq, memory, mask, pos_embed, input_type):
        
        tgt, query_pos = self.embedding(input_seq, input_type)
        tgt = tgt.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        
        repeat_nums = 1

        if not self.training:
            repeat_nums = query_pos.shape[1]

        tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        if input_type == 'pt':
            hs = self.pt_decoder(tgt, memory.repeat(1,repeat_nums,1), memory_key_padding_mask=mask.repeat(repeat_nums,1),
                        pos=pos_embed.repeat(1,repeat_nums,1), query_pos=query_pos[:len(tgt)], tgt_mask=tgt_mask)
            pred = self.pt_pred_layer(hs[-1].transpose(0, 1))
        elif input_type == 'poly':
            hs = self.poly_decoder(tgt, memory.repeat(1,repeat_nums,1), memory_key_padding_mask=mask.repeat(repeat_nums,1),
                        pos=pos_embed.repeat(1,repeat_nums,1), query_pos=query_pos[:len(tgt)], tgt_mask=tgt_mask)
            pred = self.poly_pred_layer(hs[-1].transpose(0, 1))
        elif input_type == 'rec':
            hs = self.rec_decoder(tgt, memory.repeat(1,repeat_nums,1), memory_key_padding_mask=mask.repeat(repeat_nums,1),
                        pos=pos_embed.repeat(1,repeat_nums,1), query_pos=query_pos[:len(tgt)], tgt_mask=tgt_mask)
            pred = self.rec_pred_layer(hs[-1].transpose(0, 1))

        return pred
    
    def decode_pt_seq(self, input_seq, memory, mask, pos_embed):
        pt_seq = input_seq
        pt_probs = []
        for i in range(self.args.pt_seq_length):
            pt_hs = self.decode(pt_seq, memory, mask, pos_embed, 'pt')
            pt_out = pt_hs[:, -1, :]
            pt_out = pt_out.softmax(-1)

            if not self.args.infer_vie:
                if i % 2 == 0: # coordinate or eos
                    pt_out[:, self.num_bins:self.pt_eos_index] = 0
                    pt_out[:, self.pt_eos_index+1:] = 0
                elif i % 2 == 1: # coordinate
                    pt_out = pt_out[:, :self.num_bins]
            else:
                if i % 3 == 0: # coordinate or eos
                    pt_out[:, self.num_bins:self.pt_eos_index] = 0
                    pt_out[:, self.pt_eos_index+1:] = 0
                elif i % 3 == 1: # coordinate
                    pt_out = pt_out[:, :self.num_bins]
                elif i % 3 == 2: # classes
                    pt_out[:, :-self.args.vie_categories] = 0

            pt_prob, pt_extra_seq = pt_out.topk(dim=-1, k=1)
            if pt_extra_seq[0] == self.args.pt_eos_index:
                break
            
            pt_seq = torch.cat([pt_seq, pt_extra_seq], dim=-1)
            pt_probs.append(pt_prob)                

         # remove input prompt
        if self.args.use_char_window_prompt:
            pt_seq = pt_seq[:, 7:]
        else:
            pt_seq = pt_seq[:, 5:]

        if pt_seq.shape[1] % 2 != 0:
            pt_seq = pt_seq[:, :-1]
        
        return pt_seq[0], pt_probs

    def decode_vie_pt_poly_rec_seq(self, pt_seq, pt_probs, poly_prompt, rec_prompt, image_size, memory, mask, pos_embed):
        i = 0
        result = []
        tmp_recog = []
        tmp_rect = []
        while i < len(pt_seq):
            if pt_seq[i].item() < self.args.num_bins:
                if i + 1 <= len(pt_seq) - 1:
                    if pt_seq[i+1].item() < self.args.num_bins:
                        # decode poly
                        poly_seq = torch.cat((pt_seq[i:i+2].unsqueeze(0), poly_prompt.repeat(1, 1)), dim=-1)
                        for j in range(32):
                            poly_hs = self.decode(poly_seq, memory, mask, pos_embed, 'poly')
                            poly_out = poly_hs[:, -1, :-self.args.vie_categories]
                            poly_out = poly_out.softmax(-1)
                            poly_out = poly_out[:, :self.args.num_bins]

                            poly_prob, poly_extra_seq = poly_out.topk(dim=-1, k=1)
                            poly_seq = torch.cat([poly_seq, poly_extra_seq], dim=-1)

                        image_h, image_w = image_size
                        # remove prompt
                        poly_seq = poly_seq[0,3:35].reshape(-1,2)
                        min_x = image_w.item() * poly_seq[:,0].min().item() / self.args.num_bins
                        min_y = image_h.item() * poly_seq[:,1].min().item() / self.args.num_bins
                        max_x = image_w.item() * poly_seq[:,0].max().item() / self.args.num_bins
                        max_y = image_h.item() * poly_seq[:,1].max().item() / self.args.num_bins

                        # decode rec
                        rec_seq = torch.cat((pt_seq[i:i+2].unsqueeze(0), rec_prompt.repeat(1, 1)), dim=-1)
                        for j in range(self.args.rec_length):
                            rec_hs = self.decode(rec_seq, memory, mask, pos_embed, 'rec')
                            
                            rec_out = rec_hs[:, -1, :-self.args.vie_categories]
                            rec_out = rec_out.softmax(-1)

                            rec_out[:, :self.args.num_bins] = 0
                            rec_out[:, self.args.pt_eos_index] = 0
                            rec_out[:, self.args.poly_eos_index] = 0
                            rec_out[:, self.args.rec_eos_index+1:] = 0

                            rec_prob, rec_extra_seq = rec_out.topk(dim=-1, k=1)
                            rec_seq = torch.cat([rec_seq, rec_extra_seq], dim=-1)

                        # remove start prompt
                        rec_seq = rec_seq[0,3:]

                        recog = []
                        for k in range(len(rec_seq)):
                            if rec_seq[k] == self.args.recog_pad_index:
                                break 
                            if rec_seq[k] == self.args.rec_eos_index:
                                break
                            if rec_seq[k] == self.args.recog_pad_index - 1:
                                continue
                            
                            recog.append(self.args.chars[rec_seq[k] - self.args.num_bins])

                        recog = ''.join(recog)
                        tmp_recog.append(recog)
                        tmp_rect.append([min_x, min_y, max_x, max_y])
                        i += 2
                    else:
                        i += 1
                        continue
                else:
                    i += 1
                    continue
            else:
                result.append((' '.join(tmp_recog), self.index2class[pt_seq[i].item()], pt_probs[i].item(), tmp_rect))
                i += 1
                tmp_recog = []
                tmp_rect = []
        
        return result

    def forward(self, src, mask, pos_embed, seq):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        memory = src
        
        if self.training:
            pred_pt = self.decode(seq[0], memory, mask, pos_embed, 'pt')
            pred_poly = self.decode(seq[1], memory, mask, pos_embed, 'poly')
            pred_rec = self.decode(seq[2], memory, mask, pos_embed, 'rec')

            return pred_pt, pred_poly, pred_rec
        else:
            pt_prompt = seq[0]
            poly_prompt = seq[1]
            rec_prompt = seq[2]
            pt_seq, pt_probs = self.decode_pt_seq(pt_prompt, memory, mask, pos_embed)

            if pt_seq.numel() == 0:
                return None
                
            if self.args.infer_vie:
                image_size = seq[3]
                result = self.decode_vie_pt_poly_rec_seq(pt_seq, pt_probs, poly_prompt, rec_prompt, image_size, memory, mask, pos_embed)
                return result
            else:
                # decode polygons
                pt_seq = pt_seq.reshape(-1, 2)
                pred_text_nums = pt_seq.shape[0]
                print('pred_text_nums', pred_text_nums)
                poly_seq = torch.cat((pt_seq, poly_prompt.repeat(pred_text_nums, 1)), dim=-1)
                poly_probs = []
                for i in range(32):
                    poly_hs = self.decode(poly_seq, memory, mask, pos_embed, 'poly')
                    
                    poly_out = poly_hs[:, -1, :]
                    poly_out = poly_out.softmax(-1)
                    poly_out = poly_out[:, :self.num_bins]

                    poly_prob, poly_extra_seq = poly_out.topk(dim=-1, k=1)
                    poly_seq = torch.cat([poly_seq, poly_extra_seq], dim=-1)
                    poly_probs.append(poly_prob) 

                poly_seq = poly_seq[:,3:35]

                # decode rec
                rec_seq = torch.cat((pt_seq, rec_prompt.repeat(pred_text_nums, 1)), dim=-1)
                rec_probs = []
                for i in range(self.args.rec_length):
                    rec_hs = self.decode(rec_seq, memory, mask, pos_embed, 'rec')
                    
                    rec_out = rec_hs[:, -1, :]
                    rec_out = rec_out.softmax(-1)
                    rec_out[:, :self.num_bins] = 0
                    rec_out[:, self.args.pt_eos_index] = 0
                    rec_out[:, self.args.poly_eos_index] = 0
                    rec_out[:, self.args.rec_eos_index+1:] = 0

                    rec_prob, rec_extra_seq = rec_out.topk(dim=-1, k=1)
                    rec_seq = torch.cat([rec_seq, rec_extra_seq], dim=-1)
                    rec_probs.append(rec_prob)        

                rec_seq = rec_seq[:,3:].unsqueeze(0)

                return [pt_seq.reshape(1,-1), poly_seq.reshape(1,-1), rec_seq], [torch.cat(rec_probs, dim=-1)]


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        self.pt_position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.poly_position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.rec_position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.other_position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_type):
        input_shape = x.size()
        seq_length = input_shape[1]
        batch_size = input_shape[0]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        
        input_word_embeddings = []
        input_pos_embeddings = []

        input_embeds = self.word_embeddings(x)

        if input_type == 'pt':
            position_embeds = self.pt_position_embeddings(position_ids)
        elif input_type == 'poly':
            position_embeds = self.poly_position_embeddings(position_ids)
        elif input_type == 'rec':
            position_embeds = self.rec_position_embeddings(position_ids)

        position_embeds = position_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, position_embeds


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))

        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    max_position_embeddings = 1024
    return Transformer(
        d_model=args.tfm_hidden_dim,
        nhead=args.tfm_nheads,
        num_decoder_layers=args.tfm_dec_layers,
        dim_feedforward=args.tfm_dim_feedforward,
        dropout=args.tfm_dropout,
        normalize_before=args.tfm_pre_norm,
        pad_token_id=args.padding_index,
        num_classes=args.num_classes,
        max_position_embeddings=max_position_embeddings,
        return_intermediate_dec=False,
        num_bins=args.num_bins,
        pt_eos_index=args.pt_eos_index,
        args=args,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
