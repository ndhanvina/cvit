from PIL import Image
from torchvision import transforms
import torch

"""
Loads and preprocesses an image to be compatible with the PatchEmbedding model.
    
    Args:
        image_path (str): Path to the image file.
        img_size (int): Size to which the image will be resized (default is 224x224).
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension.
        
"""

def load_image(image_path,img_size = 224):
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor