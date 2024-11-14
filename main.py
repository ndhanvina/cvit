import torch
from cvit_model.convolutional_token_embedding import TransformerEmbeddingModel
from PIL import Image
from torchvision import transforms
from helper.display_image import display_image

def main(image_path):
    # Open the image and convert it to RGB if necessary
    image = Image.open(image_path).convert('RGB')  # Ensure the image is RGB
    display_image(image)

    # Define the transformations: resize, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit the model input
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Apply the transformations and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    # Initialize the model
    model = TransformerEmbeddingModel()

    # Set the model to evaluation mode (important if using dropout or batch normalization)
    model.eval()

    # # Pass the image tensor through the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image_tensor)

    # Print the output shape
    print("Output shape:", output.shape)  # Should show the shape of the final token embeddings

if __name__ == "__main__":
    image_path = r"C:\Users\Geekonomy\Downloads\dhanvina_Profile.png"
    main(image_path)
