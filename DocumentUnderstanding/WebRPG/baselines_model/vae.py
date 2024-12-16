import imp
import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VAE(BaseVAE):


    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 parameters_len: int,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.parameters_len = parameters_len
        self.act_func = nn.LeakyReLU(0.01)

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]
        
        self.hidden_dims = hidden_dims

        self.dimension_reduction = nn.Sequential(
            nn.Linear(input_dim, 128), self.act_func)
        self.encoder_input = nn.Sequential(
            nn.Linear(128 * parameters_len, hidden_dims[-1]), self.act_func)


        # Build Encoder
        in_dim = hidden_dims[-1]
        for h_dim in hidden_dims[::-1]:
            modules.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), self.act_func))
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*modules)

        self.enc_mu = nn.Linear(hidden_dims[0], latent_dim)
        self.enc_logvar = nn.Linear(hidden_dims[0], latent_dim)

        # Build Decoder
        in_dim = latent_dim
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), self.act_func))
            in_dim = h_dim

        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1],128*parameters_len),self.act_func)
        self.final_layer2 = nn.Linear(128 ,input_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.encoder_input(torch.flatten(input,start_dim=1))
        result = self.encoder(input)

        mu = self.enc_mu(result)
        
        log_var = self.enc_logvar(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        result = self.final_layer(result)
        result = result.view(-1, self.parameters_len,128)
        result = self.final_layer2(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def make_embedding(self, input: Tensor, **kwargs):
        embedding = F.one_hot(input,self.input_dim).float()
        embedding = self.dimension_reduction(embedding)
        return embedding

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        embedding = self.make_embedding(input)
        mu, log_var = self.encode(embedding)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 

        loss_fct = nn.CrossEntropyLoss(ignore_index=2202)
        recons_loss = loss_fct(recons.permute(0,2,1),input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + mu.shape[-1] * log_var - mu ** 2 - mu.shape[-1]*log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]