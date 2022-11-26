import copy
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.model_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout_rate, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) *
                             -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

TIMM_MODELS = [
        "deit_tiny_distilled_patch16_224", 
        'deit_small_distilled_patch16_224', 
        'deit_base_distilled_patch16_224',
        'deit_base_distilled_patch16_384']

def get_pretrained_encoder(model_name):
    import timm
    assert model_name in TIMM_MODELS, "Timm Model not found"
    model = timm.create_model(model_name, pretrained=True)
    return model

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def init_xavier(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class EncoderVIT(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        
        vit = model
        self.conv1 = vit.conv1
        self.class_embedding = vit.class_embedding
        self.positional_embedding = vit.positional_embedding
        self.ln_pre = vit.ln_pre
        self.ln_post = vit.ln_post
        self.transformer = vit.transformer
        self.proj = vit.proj
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class EncoderVIT2(nn.Module):
    def __init__(self, model_name='deit_base_distilled_patch16_384'):
        super().__init__()
        
        vit = get_pretrained_encoder(model_name)
        self.embed_dim = vit.embed_dim 
        self.patch_embed = vit.patch_embed
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        
    def forward(self, src):
        x = self.patch_embed(src)
        x = self.pos_drop(x + self.pos_embed[:, 2:]) # skip dis+cls tokens
        x = self.blocks(x)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.attn_2 = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, e_outputs, src_mask, trg_mask, trg_attn_mask):
        x2 = self.norm_1(x)

        # Decoder self-attention
        x2, _ = self.attn_1(x2, x2, x2, key_padding_mask=trg_mask, attn_mask=trg_attn_mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)

        # Encoder Decoder attention
        x2, _ = self.attn_2(x2, e_outputs, e_outputs, key_padding_mask=src_mask)
        x = x + self.dropout_2(x2)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_rate=dropout)
        self.layers = get_clones(DecoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask, trg_attn_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask, trg_attn_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, trg_vocab=18022, d_model=768, d_ff=2048, N_dec=4, heads=8, dropout=0.1, model=None):
        super().__init__()

        self.encoder = EncoderVIT(model=model)
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.decoder = Decoder(trg_vocab, d_model, d_ff, N_dec, heads, dropout)
        self.out = nn.Sequential(
            nn.Linear(d_model, trg_vocab),
            nn.LogSoftmax(dim=-1)
        )
        init_xavier(self.decoder)
        init_xavier(self.out)

    def forward(self, src, trg, src_mask, trg_mask, trg_attn_mask, *args, **kwargs):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, trg_attn_mask)
        output = self.out(d_output)
        return output
        

class Transformer2(nn.Module):
    def __init__(self, trg_vocab=18022, d_model=768, d_ff=2048, N_enc=12, N_dec=3, heads=8, dropout=0.15, pretrained_encoder=True):
        super().__init__()

        self.encoder = EncoderVIT2()
        # Override decoder hidden dim if use pretrained encoder
        d_model = self.encoder.embed_dim
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.decoder = Decoder(trg_vocab, d_model, d_ff, N_dec, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

        if pretrained_encoder:
            init_xavier(self.decoder)
            init_xavier(self.out)
        else:
            init_xavier(self)

    def forward(self, src, trg, src_mask, trg_mask, *args, **kwargs):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output