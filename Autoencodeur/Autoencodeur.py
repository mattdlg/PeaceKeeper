"""
Projet Portrait robot numérique
-------------------------------
Auto-encodeur :
    Création d'un autoencodeur.
    Celui-ci compresse des images en vecteurs lantents.
    Ensuite, les vecteurs sont décodés en images.
-------------------------------
Auteurs :
    Bel Melih Morad, Anibou Amrou et Frémaux Philippine
-------------------------------
Version :
    1.1 (13/03/2025)
"""

import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ===== 1. Custom Dataset Creation =====
class CelebADataset(Dataset):
    """
    Class CelebADataset

    Custom dataset class for loading and preprocessing images.
    """
    def __init__(self, folder, transform=None, max_images=None):
        """
        Creation of an instance of the GeneticAlgorithm class.

        Parameters
        ----------
        folder : str
            Path to the folder containing images.
        transform : torchvision.transforms.Compose, optional
            Transformations applied to each image (resizing, cropping, conversion to tensor, etc.).
        max_images : int, optional
            Maximum number of images to use. If None, all images in the folder are used.

        Attributes
        ----------
        folder : str
            Path to the image directory.
        transform : torchvision.transforms.Compose or None
            Image transformations applied during loading.
        image_files : list of str
            List of image file paths, sorted and optionally limited by `max_images`.
        """
        self.folder = folder
        self.transform = transform
        self.image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        self.image_files = self.image_files[:max_images] # If max_images is set, the list is limited to the first 'max_images' files

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns
        -------
        int
            Number of available images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Retrieves an image at the specified index.

        Parameters
        ----------
        index : int
            Index of the image to retrieve.

        Returns
        -------
        torch.Tensor
            The transformed image as a tensor.
        int
            Dummy label (0), used only for compatibility with PyTorch datasets.
        """
        image_path = self.image_files[index]

        image = Image.open(image_path).convert("RGB") # Ensure 3-channel color format
        if self.transform:
            image = self.transform(image)

        return image, 0 # The label is not relevant for autoencoders


# ===== 2. Define Image Transformations =====
image_size = (128, 128)
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # Converts image to tensor with values in [0,1]
])

# ===== 3. Load Dataset =====
data_dir = "/content/img_align_celeba"  # <-- Update this path to match your local environment
dataset = CelebADataset(folder=data_dir, transform=transform_, max_images=2000)

print(f"Total number of images used : {len(dataset)}")

# ===== 4. Split Dataset (90% Train, 10% Test) =====
total_size = len(dataset)
train_size = int(0.9 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batchSize = 32
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

#print(f"Dataset d'entraînement : {train_size} images")
#print(f"Dataset de test : {test_size} images")


# ===== 5. Define the Convolutional Autoencoder =====
class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for image reconstruction.

    The autoencoder consists of:
    - An encoder: A sequence of convolutional layers that progressively reduce spatial dimension while learning a latent representation.
    - A decoder: A sequence of transposed convolution layers that reconstruct the original image from the latent representation.

    Attributes
    ----------
    encoder : nn.Sequential
        Sequence of convolutional layers encoding the input image into a latent space.
    decoder : nn.Sequential
        Sequence of transposed convolution layers reconstructing the image from the latent space.

    Methods
    -------
    forward(x, return_latent=False)
        Applies the encoder and decoder to reconstruct an image.
    encode(x)
        Extracts the latent vector from an input image.
    decode(z)
        Reconstructs an image from a latent vector.
    """

    def __init__(self):
        """
        Initializes the autoencoder architecture.

        The encoder progressively reduces image dimensions using strided convolutions.
        The decoder reconstructs the original image using transposed convolutions.
        """
        super(ConvAutoencoder, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128x128 -> 16x64x64
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 128x8x8
            nn.ReLU(True),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 64x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 16x64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 -> 3x128x128
            nn.Sigmoid()  # Normalisation between [0,1]
        )

    def forward(self, x, return_latent=False):
        """
        Performs a forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing an image of shape [batch_size, 3, 128, 128].
        return_latent : bool, optional
            If True, returns the latent vector instead of the reconstructed image.

        Returns
        -------
        torch.Tensor
            Reconstructed image of the same shape as the input.
        """
        latent = self.encoder(x)
        if return_latent:
            return latent
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """
        Extracts the latent vector from an image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing an image of shape [batch_size, 3, 128, 128].

        Returns
        -------
        torch.Tensor
            Latent vector of shape [batch_size, 128, 8, 8].
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Reconstructs an image from a latent vector.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape [batch_size, 128, 8, 8].

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape [batch_size, 3, 128, 128].
        """
        return self.decoder(z)


# ===== 6. Model Initialization, Loss Function, and Optimizer =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)

# Define loss function : choice between MSE (L2) or L1
loss_type = 'MSE'  # or 'L1'
if loss_type == 'MSE':
    criterion = nn.MSELoss()
elif loss_type == 'L1':
    criterion = nn.L1Loss()
else:
    raise ValueError("loss_type must be 'MSE' or 'L1'.")

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Model initialized and ready for training.")

# ===== 7. Training Loop =====
"""
Training Loop for the Convolutional Autoencoder
------------------------------------------------
This section trains the autoencoder using a standard supervised learning approach.  
The model learns to reconstruct input images by minimizing the reconstruction error 
(mean squared error).

Method : 
- The model is trained over "num_epochs" iterations (40 here).
- For each epoch, it processes the entire training set in mini-batches.
- The optimizer updates the weights.
- The loss is computed.

Training is performed on a GPU if available, otherwise on a CPU.
The model is saved after training.
"""

num_epochs = 40  # Number of epochs for training

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Accumulates total loss for the epoch

    for images, _ in train_loader:  # Ignore labels as it's an autoencoder
        images = images.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, images)  # Compute loss
        loss.backward()  # Backpropagation (gradient calculation)
        optimizer.step()  # Update model weights
        running_loss += loss.item() * images.size(0)  # Accumulate loss

    # Compute average loss for the epoch
    epoch_loss = running_loss / train_size
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

    # ----- Evaluation on the Test Set -----
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient calculations
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item() * images.size(0)

    # Compute average test loss for the epoch
    test_loss = test_loss / test_size
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss Test: {test_loss:.4f}")

print("Training completed successfully")

# ===== 7.1. Save Trained Model =====
torch.save(model.state_dict(), 'conv_autoencoder.pth')
print("Model saved as 'conv_autoencoder.pth'")


# ===== 8. Testing on an External Image =====
# Specify the path to the external image (ensure it is not in the training or test set)
external_image_path = "/content/img_align_celeba/002001.jpg"  # <-- Modify if necessary

# Load and preprocess the external image
external_img = Image.open(external_image_path).convert("RGB")
external_img_transformed = transform_(external_img)

# Add a batch dimension to match PyTorch model input requirements
external_img_batch = external_img_transformed.unsqueeze(0).to(device)

# Run the model in evaluation mode
model.eval()
with torch.no_grad():
    reconstructed_batch = model(external_img_batch)

# Remove batch dimension and convert tensor to image format
reconstructed_img_tensor = reconstructed_batch.squeeze(0).cpu() # Convert from [C, H, W] to [H, W, C]
reconstructed_img = reconstructed_img_tensor.numpy().transpose(1, 2, 0)

# ===== 9. Visualization of the Original and Reconstructed Image =====
plt.figure(figsize=(10, 5))

# Display the original external image
plt.subplot(1, 2, 1)
plt.title("Original external image")
plt.imshow(external_img)
plt.axis("off")

# Display the reconstructed image
plt.subplot(1, 2, 2)
plt.title("Reconstructed image")
plt.imshow(reconstructed_img)
plt.axis("off")

plt.show()


# ===== 10. Test on an external image by getting first latent vector then reconstruction =====
# Specify the path to the external image (ensure it is not in the training or test set)
external_image_path = "/content/img_align_celeba/002001.jpg"  # <-- Modify if necessary

# Load and preprocess the external image
external_img = Image.open(external_image_path).convert("RGB")
external_img_transformed = transform_(external_img)

# Add a batch dimension to match PyTorch model input requirements
external_img_batch = external_img_transformed.unsqueeze(0).to(device)

# Run the model in evaluation mode
model.eval()

# Obtain a latent vector from an image
with torch.no_grad():
    latent_vector = model.encode(external_img_batch)

print("Latent vector shape:", latent_vector.shape)

# Reconstruct image
with torch.no_grad():
    reconstructed_batch = model.decode(latent_vector)

# Remove batch dimension and convert tensor to image format
reconstructed_img_tensor = reconstructed_batch.squeeze(0).cpu() # Convert from [C, H, W] to [H, W, C]
reconstructed_img = reconstructed_img_tensor.numpy().transpose(1, 2, 0)

# ===== 11. Visualization of the Original and Reconstructed Image =====
plt.figure(figsize=(10, 5))

# Display the original external image
plt.subplot(1, 2, 1)
plt.title("Original external image")
plt.imshow(external_img)
plt.axis("off")

# Display the reconstructed image
plt.subplot(1, 2, 2)
plt.title("Reconstructed image")
plt.imshow(reconstructed_img)
plt.axis("off")

plt.show()


