import torch
import torch.nn as nn
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, CIFAR100

# Configure CUDA memory management
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model parameters with reduced memory footprint
label_max_len = 20
noise_dim = 256  # Reduced from 512
image_dim = 128 * 128 * 3  # Reduced resolution
hidden_dim = 512  # Reduced from 1024
batch_size = 8  # Reduced from 16
epochs = 500
lr = 0.0001
weight_decay = 0.0001

# Data augmentation with reduced resolution
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced from 256x256
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Choose from popular pre-built datasets with labels
dataset_name = 'CIFAR100'  # Options: 'CIFAR10', 'CIFAR100'

if dataset_name == 'CIFAR10':
    dataset = CIFAR10(
        root='./data_CIFAR10',
        download=True,
        transform=transform
    )
elif dataset_name == 'CIFAR100':
    dataset = CIFAR100(
        root='./data_CIFAR100',
        download=True,
        transform=transform
    )

# Create dataloader with reduced memory usage
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# Print dataset information
print(f"Using dataset: {dataset_name}")
print(f"Number of classes: {len(dataset.classes)}")
print(f"Class names: {dataset.classes}")
print(f"Number of images: {len(dataset)}")

# Get sample data to show details
sample_image, sample_label = dataset[0]
print(f"\nSample image details:")
print(f"  Image shape: {sample_image.shape}")
print(f"  Number of channels: {sample_image.shape[0]}")
print(f"  Image height: {sample_image.shape[1]}")
print(f"  Image width: {sample_image.shape[2]}")
print(f"  Data type: {sample_image.dtype}")
print(f"  Min value: {sample_image.min().item():.4f}")
print(f"  Max value: {sample_image.max().item():.4f}")
print(f"  Mean value: {sample_image.mean().item():.4f}")
print(f"  Label: {sample_label}")
print(f"  Label type: <class 'int'>")

# Print dataloader details
print(f"\nDataloader details:")
print(f"  Batch size: {dataloader.batch_size}")
print(f"  Number of batches: {len(dataloader)}")
print(f"  Number of workers: {dataloader.num_workers}")
print(f"  Pin memory: {dataloader.pin_memory}")

class Generator(nn.Module):
    def __init__(self, noise_dim=256, hidden_dim=512):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.variation_scale = 0.1
        
        # Simplified character embedding
        self.char_embedding = nn.Sequential(
            nn.Embedding(128, hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        # Reduced generator architecture
        self.block1 = self.generator_block(noise_dim + hidden_dim // 2, hidden_dim)
        self.block2 = self.generator_block(hidden_dim, hidden_dim // 2)
        self.block3 = self.generator_block(hidden_dim // 2, hidden_dim // 2)
        self.block4 = self.generator_block(hidden_dim // 2, hidden_dim // 4)
        
        # Final block with reduced dimensions
        self.final_block = nn.Sequential(
            nn.Linear(hidden_dim // 4, 128 * 128 * 3),  # Removed spectral norm
            nn.Tanh()
        )

    def generator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension),  # Removed spectral norm
            nn.BatchNorm1d(out_dimension),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )

    def process_label(self, labels, device):
        batch_size = len(labels)
        max_len = max(len(str(label)) for label in labels)
        
        ascii_tensor = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, label in enumerate(labels):
            ascii_chars = [ord(c) for c in str(label)]
            ascii_tensor[i, :len(ascii_chars)] = torch.tensor(ascii_chars, device=device)
        
        char_embeddings = self.char_embedding(ascii_tensor)
        return char_embeddings.mean(dim=1)

    def add_variation(self, noise):
        variation = torch.randn_like(noise) * self.variation_scale
        return noise + variation

    def forward(self, noise, labels):
        noise = self.add_variation(noise)
        label_embedding = self.process_label(labels, noise.device)
        combined = torch.cat((noise, label_embedding), 1)
        x = self.block1(combined)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        img = self.final_block(x)
        return img.view(-1, 3, 128, 128)  # Reduced resolution

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # Match generator's embedding dimension
        self.char_embedding = nn.Sequential(
            nn.Embedding(128, hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        # Image processing block with reduced dimensions
        self.img_block = self.discriminator_block(3 * 128 * 128, hidden_dim)  # Reduced resolution
        
        # Discrimination blocks with reduced dimensions
        self.discrim_block1 = self.discriminator_block(hidden_dim + hidden_dim // 2, hidden_dim // 2)
        self.discrim_block2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),  # Removed spectral norm
            nn.Sigmoid()
        )

    def discriminator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension),  # Removed spectral norm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )

    def process_label(self, labels, device):
        batch_size = len(labels)
        max_len = max(len(str(label)) for label in labels)
        
        ascii_tensor = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, label in enumerate(labels):
            ascii_chars = [ord(c) for c in str(label)]
            ascii_tensor[i, :len(ascii_chars)] = torch.tensor(ascii_chars, device=device)
        
        char_embeddings = self.char_embedding(ascii_tensor)
        return char_embeddings.mean(dim=1)

    def forward(self, image, labels):
        image = image.view(-1, 3 * 128 * 128)  # Reduced resolution
        img_features = self.img_block(image)
        label_features = self.process_label(labels, image.device)
        combined = torch.cat((img_features, label_features), 1)
        x = self.discrim_block1(combined)
        return self.discrim_block2(x).squeeze()

save_folder = './GAN'
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')

# Initialize models with memory optimization
with torch.cuda.device(device):
    gen = Generator(noise_dim=noise_dim, hidden_dim=hidden_dim).to(device)
    gen.apply(weights_init)
    disc = Discriminator(hidden_dim=hidden_dim).to(device)
    disc.apply(weights_init)

try:
    gen.load_state_dict(torch.load(os.path.join(save_folder, "Generator.pth")))
    disc.load_state_dict(torch.load(os.path.join(save_folder, "Discriminator.pth")))
except FileNotFoundError:
    print("Model files not found. Starting from scratch.")

criterion = nn.BCELoss()

# Use matching optimizer parameters for both networks
optimizer_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
optimizer_D = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
