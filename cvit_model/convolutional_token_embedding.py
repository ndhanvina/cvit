import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size
        assert height % patch_size == 0 and width % patch_size == 0, "Image size must be divisible by patch size"
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().reshape(batch_size, channels, -1, patch_size * patch_size)  # Flatten patches
        patches = patches.permute(0, 2, 3, 1)  # (batch_size, num_patches, patch_size*patch_size, channels)
        patches = patches.reshape(batch_size, -1, patch_size * patch_size * channels)  # Flatten to (batch, num_patches, embedding)
        return patches

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, patches):
        return self.linear(patches)

class ConvolutionalEmbedding(nn.Module):
    def __init__(self, embed_dim, conv_out_channels=768, kernel_size=3, stride=1):
        super(ConvolutionalEmbedding, self).__init__()
        self.conv = nn.Conv2d(embed_dim, conv_out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, patches):
        batch_size, num_patches, embed_dim = patches.shape
        patch_grid_size = int(num_patches ** 0.5)  # Assuming num_patches = 196, grid size = 14
        patches = patches.reshape(batch_size, patch_grid_size, patch_grid_size, embed_dim)
        patches = patches.permute(0, 3, 1, 2)  # (batch_size, embed_dim, grid_size, grid_size)
        patches = self.conv(patches)
        patches = patches.reshape(batch_size, -1, patches.shape[1])  # Ensure patch grid shape remains (196, 768)
        return patches

class TransformerEmbeddingModel(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, conv_out_channels=768):
        super(TransformerEmbeddingModel, self).__init__()
        self.patch_extractor = PatchExtractor(patch_size)
        self.patch_embedding = PatchEmbedding(patch_size * patch_size * 3, embed_dim)  # Use embed_dim = 768
        # self.conv_embedding = ConvolutionalEmbedding(embed_dim, conv_out_channels)  # Keep same channels in convolution

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        patches = self.patch_extractor(x)  # Extract patches from image
        patches = self.patch_embedding(patches)  # Embed patches
        # patches = self.conv_embedding(patches)  # Apply convolutional embedding to capture context
        return patches

