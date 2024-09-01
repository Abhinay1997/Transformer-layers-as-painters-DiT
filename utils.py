import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_image_grid(image_data, rows=5, cols=2, figsize=(10, 25)):
    """
    Plots an image grid with configurable rows and columns.
    
    Parameters:
    - image_data: List of tuples (title, PIL.Image)
    - rows: Number of rows in the grid (default is 5)
    - cols: Number of columns in the grid (default is 2)
    - figsize: Size of the figure (default is (10, 25))
    """
    
    # Create a figure and axis array
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each image and title
    for ax, (title, img) in zip(axes, image_data):
        ax.imshow(img)
        # ax.set_title(title)
        ax.axis('off')  # Hide the axes

    # Hide any unused subplots if there are fewer images than grid spaces
    for i in range(len(image_data), len(axes)):
        axes[i].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

# tensor = steps x layers x batch x a x b
def cosine_similarity(tensor, compute_mean=False):
    tensor = tensor.to("cuda")

    if compute_mean:
        # Step 1: Compute the average along the 's' dimension -> shape becomes (l, b, a, c)
        averaged_tensor = einops.reduce(tensor, 's l b a c -> l b a c', 'mean')
    else:
        averaged_tensor = tensor
    # Step 2: Reshape (b, a, c) into a single dimension -> shape becomes (l, a*b*c)
    reshaped_tensor = einops.rearrange(averaged_tensor, 'l a b c -> l (a b c)')

    # Step 3: Compute cosine similarity between each pair in the 'l' dimension
    # Cosine similarity: (x * y) / (||x|| * ||y||)
    # Here, we compute pairwise cosine similarities using PyTorch's F.cosine_similarity
    layers = reshaped_tensor.size()[0]
    # Initialize a similarity matrix
    cosine_similarities = torch.zeros((layers, layers))

    # Compute cosine similarity between each pair of tensors in the 'l' dimension
    for i in range(layers):
        for j in range(layers):
            cosine_similarities[i, j] = F.cosine_similarity(
                reshaped_tensor[i], reshaped_tensor[j], dim=0
            )
    return cosine_similarities

def plot_similarity_matrix(tensor):
    plt.figure(figsize=(4, 4))
    plt.imshow(tensor, cmap='viridis')
    plt.colorbar()
    plt.show()

def plot_similarity_matrices(tensors, titles):
    # Determine the grid size based on the number of tensors
    num_tensors = len(tensors)
    cols = min(3, num_tensors)  # Set max columns to 3
    rows = (num_tensors + cols - 1) // cols  # Calculate the number of rows

    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, (tensor, title) in enumerate(zip(tensors, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(tensor, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        
        y_ticks = np.arange(0, tensor.shape[0], 2)

        plt.yticks(y_ticks)

    plt.tight_layout()
    plt.show()