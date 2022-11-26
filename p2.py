import os
import glob
import json
import random
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from tokenizers import Tokenizer

import clip
from models import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# set seed
seed = 4956238
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ImgDataset(Dataset):
    def __init__(self, path, tfm=None):
        super(ImgDataset, self).__init__()
        self.path = path
        self.files = sorted(glob.glob(os.path.join(path, '*')))
        self.transform = tfm
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        file_name = self.files[idx]
        img = Image.open(file_name).convert('RGB')
        img = self.transform(img)
        return img


def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, default='hw3/p2_data/images/test/')
    parser.add_argument('--model_weight', type=str, default='model.pth')
    parser.add_argument('--output_file', type=str, default='result.json')
    parser.add_argument('--tokenized_file', type=str, default='caption_tokenizer.json')
    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=256)
    cfg = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(cfg.tokenized_file)

    print('device:', device)
    model_vit, preprocess = clip.load("ViT-L/14", device=device)
    model = Transformer(model=model_vit.visual).to(device)
    model.load_state_dict(torch.load(cfg.model_weight, map_location=device))
    model.type(torch.FloatTensor).to(device)
    model.eval()

    val_set = ImgDataset(cfg.test_folder, tfm=preprocess)
    files_name = val_set.files
    files_name = [f.split('/')[-1].split('.')[0] for f in files_name]
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    
    result = {}
    for i, batch in enumerate(tqdm(val_loader)):
        img = batch.to(device)
        x_patch = torch.ones(cfg.batch_size, cfg.max_len).to(device)
        memory = model.encoder(img)
        ys = torch.zeros(cfg.batch_size, 1).long().to(device)
        ys[:] = 2
        ans = []
        for j in range(cfg.seq_len-1):
            y_patch = torch.zeros(1, ys.shape[1]).to(device)
            with torch.no_grad():
                out = model.decoder(ys, memory, x_patch, y_patch, subsequent_mask(ys.shape[1]).to(device))
                out = model.out(out[:, -1])
            next_word = out.argmax(dim=-1).item()
            ans.append(next_word)
            ys = torch.cat([ys, torch.empty(1, 1).type_as(ys.data).fill_(next_word)], dim=1).to(device)
            if next_word == 13:
                break
        result[str(files_name[i])] = tokenizer.decode(ans)

    json_object = json.dumps(result, indent=4)
    with open(cfg.output_file, "w") as outfile:
        outfile.write(json_object)


