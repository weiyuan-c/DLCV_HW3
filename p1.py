import os
import glob
import random
import argparse

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# setting seed
seed = 453543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Imgdata(Dataset):
    def __init__(self, root, tfm=None):
        super(Imgdata, self).__init__()
        self.fnames = sorted(glob.glob(os.path.join(root, '*.png')))
        self.tfm = tfm
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname).convert('RGB')
        img = self.tfm(img)
        return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='test')
    parser.add_argument('--json', type=str, default='id2label.json')
    parser.add_argument('--predict', type=str, default='pred.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    arg = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model, preprocess = clip.load("ViT-B/32")
    model.to(device)
    model.eval()

    with open(arg.json, 'r') as f:
        data = json.load(f)
    objects = []
    for i in range(len(data)):
        objects.append(data[f'{i}'])

    prompt_text = [f'This is a photo of a {object}' for object in objects] 
    text_tokens = clip.tokenize(prompt_text).to(device)

    dataset = Imgdata(arg.folder, tfm=preprocess)
    dataloader = DataLoader(dataset, batch_size=arg.batch_size, shuffle=False)

    val_pred = []
    for i, batch in enumerate(tqdm(dataloader)):
        imgs = batch.to(device)
        with torch.no_grad():
            imgs = model.encode_image(imgs).float().to(device)
            text_features = model.encode_text(text_tokens).float().to(device)

        imgs /= imgs.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = torch.mm(text_features, imgs.T)
        pred = list(similarity.argmax(dim=0).squeeze().detach().cpu().numpy())
        val_pred += pred


    df = pd.DataFrame()
    ids = sorted([x.split('/')[-1] for x in glob.glob(os.path.join(arg.folder, '*.png'))])
    df['filename'] = np.array(ids)
    df['label'] = val_pred
    df.to_csv(arg.predict, index=False)
