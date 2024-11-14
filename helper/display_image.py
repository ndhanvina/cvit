import matplotlib.pyplot as plt

def display_image(image):
    """Function to display the image."""
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()