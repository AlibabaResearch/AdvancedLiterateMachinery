import torch
import os
import copy

from baselines_model.vae import VAE
from markuplm import MarkupLMConfig, MarkupLMModel
from baselines_model.FID_model import FIDBackbone, FIDWebModel
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm 
import random
import argparse
import json
import numpy as np


random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def matrix_sqrt(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    sqrt_eigenvalues = np.sqrt(np.abs(eigenvalues))
    sqrt_matrix = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)
    return sqrt_matrix

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)

    sqrt_sigma1 = matrix_sqrt(sigma1)
    sqrt_sigma2 = matrix_sqrt(sigma2)
    covmean = sqrt_sigma1 @ sqrt_sigma2
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0*covmean)
    fid = fid.real
    return fid


def load_pretrained_model(model, path, map_location='cpu'):
    model_CKPT = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location)
    model.load_state_dict(model_CKPT, strict=False)
    return model

def build_vae(input_dim=1993, latent_dim=128, layer_num=5, start_hidden_dim=128, parameters_len=13):
    hidden_dims = [start_hidden_dim * (2 ** i) for i in range(layer_num)]
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, parameters_len=parameters_len,hidden_dims=hidden_dims)
    return vae

def build_markuplm(path):
    config = MarkupLMConfig.from_pretrained(path)
    model = MarkupLMModel(config)
    return load_pretrained_model(model, path)

def build_fid_web_model(vae, xpath_layer):
    backbone = FIDBackbone(
        in_dim=128,
        out_dim=2,
        embed_dim=128,
        depth=4,
        chrlen_dim=128,
        xpath_dim=1024,
        num_element_tokens=512
    )
    model = FIDWebModel(
        backbone,
        max_chrlen=512,
        chrlen_dim=128,
        vae=copy.deepcopy(vae),
        xpath_layer=copy.deepcopy(xpath_layer)
    )
    return model


class MyDataset(Dataset):
    def __init__(self, pt_dir):
        self.pt_dir = pt_dir
        self.file_lis = os.listdir(pt_dir)

    def __len__(self):
        return len(self.file_lis)

    def __getitem__(self, idx):
        file_name = self.file_lis[idx]
        data = torch.load(os.path.join(self.pt_dir, file_name))
        return data

def get_embeddings(disc_model, loader, fid_type="overall", pad_value=1992):
    gt_act = []
    pred_act = []
    pred_logist = []
    
    for batch in tqdm(loader):
        # Unpack data
        batch_size = batch["element_mask"].shape[0]
        element_mask = batch["element_mask"].to(disc_model.device)
        all_xpath_tags_seq = batch["all_xpath_tags_seq"].to(disc_model.device)
        all_xpath_subs_seq = batch["all_xpath_subs_seq"].to(disc_model.device)
        chrlen = batch["chrlen"].to(disc_model.device)
        param = batch["pred"].to(disc_model.device)

        if fid_type == "layout":
            param[:,:,4:] = pad_value
        if fid_type == "style":
            param[:,:,:4] = pad_value

        perturb = torch.ones(batch_size).to(disc_model.device)
    
        # Prediction Embed
        with torch.no_grad():
            pred_embed = disc_model(param=param, perturb=perturb, all_xpath_tags_seq=all_xpath_tags_seq,
                                    all_xpath_subs_seq=all_xpath_subs_seq, chrlen=chrlen, element_mask=element_mask)
            pred_act.append(pred_embed["hidden_state"])
            pred_logist.append(pred_embed["output"])

        # Ground Truth Embed
        param = batch["gt"].to(disc_model.device)

        if fid_type == "layout":
            param[:,:,4:] = pad_value
        if fid_type == "style":
            param[:,:,:4] = pad_value

        perturb = torch.zeros(batch_size).to(disc_model.device)
        with torch.no_grad():
            gt_embed = disc_model(param=param, perturb=perturb, all_xpath_tags_seq=all_xpath_tags_seq,
                                  all_xpath_subs_seq=all_xpath_subs_seq, chrlen=chrlen, element_mask=element_mask)
            gt_act.append(gt_embed["hidden_state"])

    return torch.cat(gt_act, dim=0).cpu().numpy(), torch.cat(pred_act, dim=0).cpu().numpy()

def setup_fid_web_model(pretrained_vae, xpath_layer, pretrained_model_path):
    disc_model = build_fid_web_model(pretrained_vae, xpath_layer)
    disc_model = load_pretrained_model(disc_model, pretrained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc_model.to(device)
    disc_model.eval()
    
    return disc_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate FID.')
    parser.add_argument('--pt_dir', type=str, help='Path to PT directory')
    parser.add_argument('--fid_type', type=str, choices=['overall', 'style', 'layout'], help='Type of FID')
    
    args = parser.parse_args()

    print("Start calculating FID")
    pt_dir = args.pt_dir
    fid_type = args.fid_type
    pretrained_model_path = {
        "overall": "./FID_model/overall",
        "style": "./FID_model/style",
        "layout": "./FID_model/layout"
    }
    pretrained_model_path = pretrained_model_path[fid_type]

    pretrained_vae = build_vae()

    markuplm = build_markuplm("../css_data/markuplm-large")
    xpath_layer = copy.deepcopy(markuplm.embeddings.xpath_embeddings)


    dataset = MyDataset(pt_dir)
    loader = DataLoader(dataset, batch_size=128, num_workers=8) 

    disc_model = setup_fid_web_model(pretrained_vae, xpath_layer, pretrained_model_path)

    gt, pred = get_embeddings(disc_model,loader,fid_type)
    fid = calculate_fid(gt, pred)
    print(f"FID_{fid_type}: {fid}")