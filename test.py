from data import torch, batch_size, noise_dim, device, gen, plt
import numpy as np

def get_noise(n_samples, noise_dim, device='cpu'):
    """Generate random noise vectors."""
    return torch.randn(n_samples, noise_dim, device=device)

def generate_images(num_samples=4, noise_dim=256, image_size=(128, 128)):
    """
    Generate images using the generator model with user-provided label.
    Returns a grid of images for visualization.
    """
    # Generate noise vectors
    noise = get_noise(num_samples, noise_dim, device)
    
    # Get user input for label
    user_label = input('Input the Quarry: ')
    test_labels = [user_label] * num_samples
    
    # Generate images using the generator
    with torch.no_grad():
        fake_images = gen(noise, test_labels).cpu()
    
    # Denormalize images (assuming images are normalized between -1 and 1)
    fake_images = (fake_images + 1) / 2.0  # Scale to [0, 1]
    
    # Clip values to ensure they are in [0, 1]
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # Plot the images in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        # Reshape and transpose for proper visualization (assuming RGB channels)
        image = fake_images[i].permute(1, 2, 0).numpy()  # Shape: (128, 128, 3)
        # Add slight variation to each image
        image = np.clip(image + np.random.normal(0, 0.05, image.shape), 0, 1)
        ax.imshow(image)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('out.png')
    plt.close()
    
    return fake_images

# Generate and display the images
if __name__ == "__main__":
    generated_images = generate_images()
    print("Images generated and saved as 'out.png'")
