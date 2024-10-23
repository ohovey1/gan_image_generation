import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model function
def load_model():
    # Manually recreate the model based on the loaded architecture
    generator = nn.Sequential(
        nn.ConvTranspose2d(100, 256, kernel_size=(7, 7), stride=(1, 1)),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        nn.Tanh()
    )
    
    # Load the model weights
    state_dict = torch.load('generator.h5', map_location=device)
    # Remove 'model.' prefix from state_dict keys
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    generator.load_state_dict(new_state_dict)

    # Set to evaluation mode
    generator.to(device).eval()
    return generator

# Generate image function
def generate_image(model, num_images=1):
    '''
    Take the model as input and generate one image.
    '''
    # Set the dimensions of the noise
    z_dim = 100
    z = np.random.normal(size=[num_images, z_dim])
    z = torch.tensor(z, dtype=torch.float32).view(num_images, z_dim, 1, 1).to(device)
    
    # Generate an image using the trained Generator
    with torch.no_grad():
        generated_image = model(z).cpu().numpy()
    return generated_image

if __name__ == "__main__":
    model = load_model()
    images = generate_image(model, num_images=32)

    # Plot generated images
    plt.figure(figsize=(12, 8))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.axis("off")
        plt.imshow(images[i, 0, :, :], cmap='gray')
    plt.show()

