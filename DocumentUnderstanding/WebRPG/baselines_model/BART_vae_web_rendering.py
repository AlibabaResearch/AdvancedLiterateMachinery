import torch
import torch.nn as nn
import math
import torch.utils.checkpoint
import torch.nn.functional as F
from markuplm import MarkupLMConfig, MarkupLMModel
import numpy as np
import copy

class BartVAEBackbone(nn.Module):
    def __init__(self,in_dim, out_dim, embed_dim=128, 
                 norm_layer=nn.LayerNorm, num_element_tokens=None, global_text_dim=None, element_text_dim=None, chrlen_dim=None, xpath_dim=None, bart=None):
        super().__init__()
        self.out_dim = out_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.in_embed = nn.Linear(in_dim, embed_dim)

        # global text feature & element text feature
        self.global_text_embed = nn.Linear(global_text_dim, embed_dim)
        self.element_text_embed = nn.Linear(element_text_dim, embed_dim)
        # null text embed
        self.null_global_text_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        # element character length
        self.chrlen_embed = nn.Linear(chrlen_dim, embed_dim)
        # element xpath
        self.xpath_embed = nn.Linear(xpath_dim, embed_dim)
        
        self.num_element_tokens = num_element_tokens
        self.num_tokens = 1 + num_element_tokens # global text + elements

        self.bart=bart

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.out_dim, bias=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(
        self,
        param,
        global_text,
        element_text,
        chrlen,
        xpath,
        element_mask
    ): 

        device = param.device
        param_embed = self.in_embed(param)
        global_text = self.global_text_embed(global_text)
        element_text = self.element_text_embed(element_text)
        chrlen = self.chrlen_embed(chrlen)
        xpath = self.xpath_embed(xpath)
        
        # sum element features
        element_tokens = torch.zeros_like(param_embed, device=device)        
        element_tokens = param_embed + element_text + xpath + chrlen

        # concat time & global_text & element features
        x = torch.cat((global_text, element_tokens), dim=1)

        output = self.bart(decoder_inputs_embeds=x,inputs_embeds=x,attention_mask=torch.cat((torch.ones(element_mask.shape[0],1,device=device),element_mask),dim=-1))
        x = self.decoder_pred(output.last_hidden_state[:,1:,:]) #remove global text
    
        return x, output


class BartVAEWebModel(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_chrlen = 512,
        chrlen_dim = 128,
        config = None,
        xpath_layer=None,
        gamma_func="cosine",
        vae=None,
        mask_dim=128,
        kld_weight=0.001,
        vae_loss_weight=1.0,
        max_render_range=1920

    ):
        super().__init__()
        self.config = config

        # construct model
        self.model = model # the model is initialized outside

        # chrlen_embed to encode the chrlen int64 index to embedding
        self.chrlen_embed = nn.Embedding(max_chrlen,chrlen_dim)

        # xpath layer
        self.xpath_layer = xpath_layer
        
        # vae
        self.vae = vae
        self.kld_weight = kld_weight
        self.vae_loss_weight = vae_loss_weight

        self.gamma = self.gamma_func(gamma_func)
        self.mask_text_embed = nn.Parameter(torch.randn(1,1,mask_dim))

        self.max_render_range = max_render_range

        # device tracker
        self.register_buffer('_dummy', torch.Tensor([True]), persistent = False)

        self._keys_to_ignore_on_save = []

    @property
    def device(self):
        return self._dummy.device

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError 

    # Process the sequence_output along the element dimension
    def process_sequence_output(self,sequence_output,element_bos):
        masks = torch.zeros([sequence_output.shape[0],1,sequence_output.shape[-1]],device=sequence_output.device)
        sequence_output = sequence_output.scatter(dim=1,index=torch.zeros(sequence_output.shape[0],1,sequence_output.shape[-1],dtype=torch.int64,device=sequence_output.device),src=masks)
        sequence_output = sequence_output.unsqueeze(1).expand(sequence_output.shape[0],element_bos.shape[1],sequence_output.shape[1],sequence_output.shape[2])
        output = torch.gather(sequence_output,dim=2,index=element_bos.unsqueeze(-1).expand(element_bos.size(0),element_bos.size(1),element_bos.size(2),sequence_output.shape[-1]))
        count = torch.sum(output != 0, dim=2)
        count[count == 0] = 1
        output = torch.sum(output,2)/count
        return output

    def forward(
        self,
        param, 
        chrlen, 
        element_mask,
        global_text=None,
        element_text=None,
        xpath=None,
        all_xpath_tags_seq=None,
        all_xpath_subs_seq=None,
        mask_ratio=None,
        input_ids=None,
        input_mask=None,
        element_bos=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None
    ):
        model               = self.model
        channel             = param.shape[-1]
        device              = param.device
        element_len         = param.shape[1]
        batch_size          = param.shape[0]

        if global_text is not None:
            global_text = torch.tensor(global_text).type(torch.float32)
        
        if element_text is not None:
            element_text = torch.tensor(element_text).type(torch.float32)
        
        embedding = self.vae.make_embedding(param.view(-1,channel))
        mu, log_var = self.vae.encode(embedding)
        x_start = self.vae.reparameterize(mu, log_var)
        x_start = x_start.view(batch_size,-1,x_start.shape[-1])
        
        # If mask is TRUE, apply masking; if FALSE, do not apply masking.
        if mask_ratio is None:
            r = math.floor(self.gamma(np.random.uniform()) * param.shape[1])
        else:
            if isinstance(mask_ratio,torch.Tensor):
                mask_ratio = mask_ratio[0].cpu()
            r = math.floor(self.gamma(mask_ratio) * param.shape[1])
        mask_shape = (param.shape[0],param.shape[1])
        sample = torch.rand(mask_shape, device=device).topk(r, dim=1).indices
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=sample, value=True)
        mask = mask.unsqueeze(-1) 
        masked_x_start = torch.where(mask,self.mask_text_embed,x_start)

        label = torch.where(mask,param,torch.full(param.shape,-100,device=device))
        label = torch.where(element_mask.unsqueeze(-1).bool(),label,torch.full(param.shape,-100,device=device))

        if torch.all(label == -100).item():
            label = torch.where(element_mask.unsqueeze(-1).bool(),param,torch.full(param.shape,-100,device=device))
            
        chrlen = self.chrlen_embed(chrlen)

        xpath = self.xpath_layer(all_xpath_tags_seq,all_xpath_subs_seq)

        model_output,output_object = self.model(
            masked_x_start,
            global_text,
            element_text,
            chrlen,
            xpath,
            element_mask)

        # VAE loss
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        param = torch.where(element_mask.unsqueeze(-1).expand_as(param).bool(), param, -100)
        recons_loss,kld_loss,vae_loss = torch.tensor(0.0,device=device), torch.tensor(0.0,device=device), torch.tensor(0.0,device=device)
        if mu is not None and log_var is not None:
            x_start = self.vae.decode(x_start.view(-1,x_start.shape[-1]))
            recons = x_start.view(batch_size,-1,x_start.shape[-1])
            kld_weight = self.kld_weight
            
            recons_loss = ce_loss(recons.view(-1,recons.shape[-1]),param.view(-1))
            kld_loss = torch.mean(-0.5 * torch.sum(1 + mu.shape[-1] * log_var - mu ** 2 - mu.shape[-1]*log_var.exp(), dim = 1), dim = 0)
            kld_loss = kld_weight * kld_loss
            vae_loss = recons_loss + kld_loss
            vae_loss = vae_loss * self.vae_loss_weight
        
        model_output = self.vae.decode(model_output.view(-1,model_output.shape[-1]))
        model_output = model_output.view(batch_size,element_len,-1,model_output.shape[-1])

        loss = ce_loss(model_output.view(-1,model_output.shape[-1]),label.view(-1))
            

        return {
            "loss": loss + vae_loss,
            "bart_loss": loss,
            "vae_loss": vae_loss,
            "recons_loss": recons_loss,
            "kld_loss": kld_loss,
            "pred": model_output,
            "last_hidden_state": output_object.last_hidden_state,
            "encoder_last_hidden_state": output_object.encoder_last_hidden_state
        }

