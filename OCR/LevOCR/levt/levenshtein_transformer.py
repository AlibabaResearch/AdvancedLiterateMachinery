import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List
from collections import namedtuple

from levt.transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from levt.fairseq_nat_model import ensemble_decoder
from levt.fairseq_dropout import FairseqDropout
from levt.positional_embedding import PositionalEmbedding
from levt.multihead_attention import MultiheadAttention

from levt.levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _get_del_targets,
    _get_ins_targets,
    _skip,
    _skip_encoder_out,
)
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history"],
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class LevenshteinTransformerModel(nn.Module):

    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.supports_align_args = True
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()
        self.ensemble_models = None
        self.encoder = encoder
        self.decoder = decoder
        self._is_generation_fast = False

    @classmethod
    def build_model(cls, args, src_dict):
        """Build a new model instance."""
        encoder_embed_tokens = cls.build_embedding(
            src_dict, args.encoder_embed_dim
        )
        decoder_embed_tokens = encoder_embed_tokens
        args.share_decoder_input_output_embed = True

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, src_dict, decoder_embed_tokens)

        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        return emb

    @property
    def allow_length_beam(self):
        return False

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = LevenshteinTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder


    def forward(
        self, vlt_input, img_feature, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."
        
        rand_tokens = vlt_input.to(tgt_tokens.device)     
        
        text_feature, _ = self.encoder.forward_feature(
            normalize=False,
            prev_output_tokens=rand_tokens,
            encoder_out=None,
        )
        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            rand_tokens, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = rand_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            img_feature=img_feature,
            text_feature=text_feature,
        )
        mask_ins_out = mask_ins_out[:,:rand_tokens.size(1)-1,:]

        text_feature_maskin, _ = self.encoder.forward_feature(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=None,
        )
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            img_feature=img_feature,
            text_feature=text_feature_maskin,
        )
        word_ins_out = word_ins_out[:,:rand_tokens.size(1),:]

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
            ).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        text_feature_wordin, _ = self.encoder.forward_feature(
            normalize=False,
            prev_output_tokens=word_predictions,
            encoder_out=None,
        )

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            img_feature=img_feature,
            text_feature=text_feature_wordin,
        )
        word_del_masks = word_predictions.ne(self.pad)
        word_del_out = word_del_out[:,:rand_tokens.size(1),:]           

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
                "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
        }
      
    def forward_text(
        self, ctc_result, img_feature, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."
        
        text_levt_add = ctc_result[2].to(tgt_tokens.device)      
        
        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            text_levt_add, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = text_levt_add[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.encoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=text_levt_add,
            encoder_out=None,
        )
        # mask_ins_out = mask_ins_out[:,:prev_output_tokens.size(1)-1,:]

        word_ins_out, _ = self.encoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=None,
        )
        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
            ).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.encoder.forward_word_del(
            normalize=False,
            prev_output_tokens=word_predictions,
            encoder_out=None,
        )
        word_del_masks = word_predictions.ne(self.pad)

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
                "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
        }
        
    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)


    def forward_decoder(
        self, decoder_out, img_feature, eos_penalty=0.0, max_ratio=None, **kwargs
    ):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        max_src_len = img_feature.size(1)
        src_lens = img_feature.new(bsz).fill_(max_src_len)
        max_lens = (src_lens * max_ratio).clamp(min=10).long()

        del_text = output_tokens
        text_feature, _ = self.encoder.forward_feature(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=None,
        )
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                img_feature=img_feature,
                text_feature=text_feature,
            )
            
            word_del_pred = word_del_score.max(-1)[1].bool()
            tmp_score = F.softmax(word_del_score,-1)
            word_del_pred = tmp_score[:,:,1]>self.args.th

            word_del_pred = word_del_pred[:,:output_tokens.size(1)]
            word_del_score = word_del_score[:,:output_tokens.size(1),:]

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens,
                output_scores,
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)
        history.append(output_tokens.clone())

        if torch.all(torch.eq(del_text, output_tokens)):
            text_feature_maskin = text_feature
        else:
            text_feature_maskin, _ = self.encoder.forward_feature(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=None,
            )
        maskin_text = output_tokens
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                img_feature=img_feature,
                text_feature=text_feature_maskin,
            )
            
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )
            mask_ins_pred = mask_ins_pred[:,:output_tokens.size(1)-1]
            mask_ins_score = mask_ins_score[:,:output_tokens.size(1)-1,:]

            _tokens, _scores = _apply_ins_masks(
                output_tokens,
                output_scores,
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            
            try:
                output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
                output_scores = _fill(output_scores, can_ins_mask, _scores, 0)
            except:
                output_tokens = _tokens
                output_scores = _scores
        history.append(output_tokens.clone())
        
        if torch.all(torch.eq(maskin_text, output_tokens)):
            text_feature_wordin = text_feature_maskin
        else:
            text_feature_wordin, _ = self.encoder.forward_feature(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=None,
            )

        #can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        can_ins_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                img_feature=img_feature,
                text_feature=text_feature_wordin,
            )
            word_ins_score = word_ins_score[:,:output_tokens.size(1),:]
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            word_ins_pred = word_ins_pred[:,:output_tokens.size(1)]

            _tokens, _scores = _apply_ins_words(
                output_tokens,
                output_scores,
                word_ins_pred,
                word_ins_score,
                self.unk,
            )
            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)
        history.append(output_tokens.clone())
        
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history
        )

    def initialize_output_tokens(self, vlt_input):
        initial_output_tokens = vlt_input.to(torch.long)      
        initial_output_scores = torch.zeros_like(initial_output_tokens).to(initial_output_tokens.device)
        initial_output_scores = initial_output_scores.to(torch.float)  

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=[]
        )


class LevenshteinTransformerEncoder(nn.Module):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__()
        self.ensemble_models = None

        self.output_embed_dim = args.decoder_output_dim
        self.args = args
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_encoder_layer(args, no_encoder_attn)
                for _ in range(int(args.decoder_layers))
            ]
        )
        self.num_layers = len(self.layers)
        self.adaptive_softmax = None
        self.output_projection = None
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

        # call FairseqDecoder
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None        

        self.dictionary = dictionary
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        self.layers_del = None

        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=512 * 4,
            max_position_embeddings=len(self.dictionary),
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)
        self.token_type_embeddings = nn.Embedding(2, 512)
        self.token_type_embeddings.apply(init_weights)  
        self.embed_dim = (
            Linear(512, embed_dim, bias=False)
        )          

    def build_output_projection(self, args, dictionary, embed_tokens):
        self.output_projection = nn.Linear(
            self.embed_tokens.weight.shape[1],
            self.embed_tokens.weight.shape[0],
            bias=False,
            )
        self.output_projection.weight = self.embed_tokens.weight

    def build_encoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        return layer

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """

        #print('prev_output_tokens[128, 35]:', prev_output_tokens.shape)
        text_embeds = self.text_embeddings(prev_output_tokens)
        #print('text_embeds[128, 35, 512]:', text_embeds.shape)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(prev_output_tokens))
        x = text_embeds
        text_masks = prev_output_tokens.eq(self.padding_idx)
        co_masks = text_masks

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                None,
                None,
                #x,
                #co_masks,
                self_attn_mask=None,
                self_attn_padding_mask=co_masks,
            )
            inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)

        if normalize:
            print('normalize1:', normalize)
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            print('normalize2:', normalize)
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    def forward_feature(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        return features, extra["attn"]

    def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            print('normalize3:', normalize)
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    def output_layer(self, features):
        return self.output_projection(features)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            print('len(encoder_out[encoder_out]) == 0')
            new_encoder_out = []
        else:
            print('len(encoder_out[encoder_out]) != 0')
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if not encoder_out["encoder_padding_mask"] or len(encoder_out["encoder_padding_mask"]) == 0:
            print('not encoder_out[encoder_padding_mask] or len(encoder_out[encoder_padding_mask]) == 0')
            new_encoder_padding_mask = []
        else:
            print('else: encoder_out[encoder_padding_mask] or len(encoder_out[encoder_padding_mask]) == 0')
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if not encoder_out["encoder_embedding"] or len(encoder_out["encoder_embedding"]) == 0:
            print('not encoder_out[encoder_embedding]')
            new_encoder_embedding = []
        else:
            print('else: not encoder_out[encoder_embedding]')
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            print('len(encoder_out[src_tokens]) == 0')
            src_tokens = []
        else:
            print('else: len(encoder_out[src_tokens]) == 0')
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            print('len(encoder_out[src_lengths]) == 0')
            src_lengths = []
        else:
            print('else: len(encoder_out[src_lengths]) == 0')
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if encoder_states and len(encoder_states) > 0:
            print('encoder_states and len(encoder_states) > 0')
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class LevenshteinTransformerDecoder(nn.Module):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__()
        self.ensemble_models = None

        self.output_embed_dim = args.decoder_output_dim
        self.args = args
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.quant_noise = None

        self.project_in_dim = None

        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layernorm_embedding = None
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(int(args.decoder_layers))
            ]
        )
        self.num_layers = len(self.layers)
        self.adaptive_softmax = None
        self.output_projection = None
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None        

        self.dictionary = dictionary
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        self.layers_del = None

        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=512 * 4,
            max_position_embeddings=len(self.dictionary),
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)
        self.token_type_embeddings = nn.Embedding(2, 512)
        self.token_type_embeddings.apply(init_weights)  
        self.embed_dim = (
            Linear(512, embed_dim, bias=False)
        )          

    def build_output_projection(self, args, dictionary, embed_tokens):
        self.output_projection = nn.Linear(
            self.embed_tokens.weight.shape[1],
            self.embed_tokens.weight.shape[0],
            bias=False,
        )
        self.output_projection.weight = self.embed_tokens.weight

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        return layer

    def extract_features(
        self,
        img_feature,
        text_feature,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """

        posi_token = torch.ones((text_feature.size(0), text_feature.size(1))).long().to(text_feature.device)
        text_embeds = self.text_embeddings(posi_token)
        
        text_token_type_idx = 1
        text_masks = torch.LongTensor(text_feature.size()[:2]).fill_(0).to(text_feature.device)
        text_embeds = text_feature + text_embeds + self.token_type_embeddings(torch.full_like(text_masks, text_token_type_idx))
        if img_feature != None:
            image_embeds = self.embed_dim(img_feature)
            image_token_type_idx = 1
            image_masks = torch.LongTensor(img_feature.size()[:2]).fill_(0).to(text_feature.device)
            image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))

            x = torch.cat([text_embeds, image_embeds], dim=1)
            image_masks = image_masks.type(torch.BoolTensor).to(text_feature.device)
            #text_masks = prev_output_tokens.eq(self.padding_idx)
            co_masks = torch.cat([text_masks, image_masks], dim=1).bool()
        else:
            x = text_embeds
            co_masks = text_masks

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                None,
                None,
                #x,
                #co_masks,
                self_attn_mask=None,
                self_attn_padding_mask=co_masks,
            )
            inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, img_feature, text_feature, **unused):
        features, extra = self.extract_features(
            img_feature,
            text_feature,
            early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)

        if normalize:
            # print('normalize4:', normalize)
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_word_ins(self, normalize, img_feature, text_feature, **unused):
        features, extra = self.extract_features(
            img_feature,
            text_feature,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            # print('normalize5:', normalize)
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_word_del(self, normalize, img_feature, text_feature, **unused):
        features, extra = self.extract_features(
            img_feature,
            text_feature,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            # print('normalize6:', normalize)
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    def output_layer(self, features):
        return self.output_projection(features)



