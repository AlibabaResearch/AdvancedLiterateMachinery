import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import BertConfig, BertModel

class FIDBackbone(nn.Module):
    def __init__(self,in_dim, out_dim, embed_dim=128, depth=4,
                 num_heads=8, norm_layer=nn.LayerNorm, num_element_tokens=None, chrlen_dim=None, xpath_dim=None):
        super().__init__()
        self.out_dim = out_dim
        self.num_features = self.embed_dim = embed_dim
        
        self.in_embed = nn.Linear(in_dim, embed_dim)

        # cls text embed
        self.cls_text_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.chrlen_embed = nn.Linear(chrlen_dim, embed_dim)
        self.xpath_embed = nn.Linear(xpath_dim, embed_dim)
        
        self.num_element_tokens = num_element_tokens
        self.num_tokens = 1 + num_element_tokens # cls + elements

        bert_config = BertConfig(
            max_position_embeddings=self.num_tokens,
            num_hidden_layers=depth,
            hidden_size=self.embed_dim,
            num_attention_heads=num_heads
            )
        self.bert=BertModel(bert_config)

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.out_dim, bias=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(
        self,
        param,
        chrlen,
        xpath,
        element_mask,
    ): 
        device = param.device

        param_embed = self.in_embed(param)

        chrlen = self.chrlen_embed(chrlen)
        xpath = self.xpath_embed(xpath)

        element_tokens = param_embed + chrlen + xpath

        cls_text_embed = self.cls_text_embed.to(element_tokens.dtype)
        cls_text_embed = cls_text_embed.expand(element_tokens.shape[0],cls_text_embed.shape[1],cls_text_embed.shape[2])
        x = torch.cat((cls_text_embed, element_tokens), dim=1)

        output = self.bert(inputs_embeds=x,attention_mask=torch.cat((torch.ones(element_mask.shape[0],1,device=device),element_mask),dim=-1))
        x = self.decoder_pred(output.pooler_output)
    
        return x, output.pooler_output


class FIDWebModel(nn.Module):
    def __init__(
        self,
        model,                                      
        *,
        max_chrlen = 512,
        chrlen_dim = 128,
        config = None,
        xpath_layer=None,
        vae=None
    ):
        super().__init__()

        self.config = config
    
        self.model = model # the model is initialized outside

        self.chrlen_embed = nn.Embedding(max_chrlen,chrlen_dim)

        # vae
        self.vae = vae

        # xpath layer
        self.xpath_layer = xpath_layer

        # device tracker
        self.register_buffer('_dummy', torch.Tensor([True]), persistent = False)

    @property
    def device(self):
        return self._dummy.device


    def forward(
        self,
        param, 
        chrlen,      
        element_mask,
        perturb,
        xpath=None,
        all_xpath_tags_seq=None,
        all_xpath_subs_seq=None,
        **kwargs
    ):
        model               = self.model
        channel             = param.shape[-1]
        device              = param.device
        element_len         = param.shape[1]
        batch_size          = param.shape[0]

        embedding = self.vae.make_embedding(param.view(-1,channel))
        mu, log_var = self.vae.encode(embedding)
        x_start = self.vae.reparameterize(mu, log_var)
        x_start = x_start.view(batch_size,-1,x_start.shape[-1])

        chrlen = self.chrlen_embed(chrlen)

        xpath = self.xpath_layer(all_xpath_tags_seq,all_xpath_subs_seq)

        model_output,hidden_state = self.model(
            x_start,
            chrlen,
            xpath,
            element_mask)

        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        perturb = perturb.long()
        loss = ce_loss(model_output,perturb)

        return {
            "loss":loss,
            "output":model_output,
            "hidden_state":hidden_state
        }


