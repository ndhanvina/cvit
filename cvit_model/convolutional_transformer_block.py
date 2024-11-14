import torch
import torch.nn as nn
import torch.nn.functional as F
from cvit_model.convolutional_token_embedding import PatchEmbedding

class ConvolutionalTransformerBlock(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, num_heads=8, num_layers=6, img_size=224, patch_size=16):
        super(ConvolutionalTransformerBlock, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Convolutional Layer for Feature Extraction
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        
        # Patch Embedding using a Conv Layer
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=embed_dim, embed_dim=embed_dim)
        
        # Transformer Encoder Layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Final Linear Layer for Output (e.g., for classification)
        self.fc_out = nn.Linear(embed_dim, 10)  # Assuming 10 output classes (adjust as per your task)
    
    def forward(self, x):
        # Pass through the convolutional layer
        x = self.conv(x)  # Extracts local features
        
        # Embed patches using PatchEmbedding (to create sequence of patches)
        x = self.patch_embedding(x)
        
        # Pass through transformer encoder layers
        x = self.transformer(x)
        
        # Global Average Pooling and Classification
        x = x.mean(dim=1)  # Pooling across sequence dimension (flatten all patches)
        x = self.fc_out(x)  # Classifier
        return x