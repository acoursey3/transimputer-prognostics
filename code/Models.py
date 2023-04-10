import torch
import torch.nn as nn

# Should we use a linear attention transformer?
class ImputationTransformer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.input_projection = nn.Linear(542*24, embed_dim)
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, activation="gelu")
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
    
    def forward(self, x):
        x = x.flatten(1)
        z = self.input_projection(x)
        z = z + self.positional_embed
        z = self.transformer_blocks(z)
        
        return z
    
class ReconstructionImputationTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embed_dim = 128
        self.imputation_transformer = ImputationTransformer(self.embed_dim)
        self.reconstruction = nn.Sequential(nn.Linear(self.embed_dim, 2048),
                                           nn.ReLU(),
                                           nn.Linear(2048, 542*24))
    
    def forward(self, x):
        z = self.imputation_transformer(x)
        z = self.reconstruction(z)
        
        return z.view(x.shape[0],542,24)