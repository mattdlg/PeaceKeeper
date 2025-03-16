import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ===== 1. Définition de l'architecture mise à jour =====
class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder model for image compression and reconstruction.

    Attributes
    ----------
    encoder : torch.nn.Sequential
        Sequential container of encoder layers
    decoder : torch.nn.Sequential
        Sequential container of decoder layers
    """

    def __init__(self):
        """Initialize the ConvAutoencoder with encoder and decoder layers."""

        super(ConvAutoencoder, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, return_latent=False):
        """Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 128, 128)
        return_latent : bool, optional
            Whether to return the latent vector instead of reconstruction

        Returns
        -------
        torch.Tensor
            Reconstructed image if return_latent=False, latent vector otherwise
        """
        latent_vector = self.encoder(x)
        if return_latent:
            return latent_vector
        return self.decoder(latent_vector)

    def encode(self, x):
        """Encode an input image to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 128, 128)

        Returns
        -------
        torch.Tensor
            Latent vector of shape (batch_size, 128, 8, 8)
        """
        return self.encoder(x)

    def decode(self, z):
        """Decode a latent vector to image space.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch_size, 128, 8, 8)

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (batch_size, 3, 128, 128)
        """
        return self.decoder(z)


# ===== 2. Configuration =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 3. Chargement du modèle =====
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load('conv_autoencoder.pth', map_location=device))
model.eval()

# ===== 4. Transformation des images =====
image_size = (128, 128)
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])


# ===== 5. Fonction d'inférence complète =====
def process_image(img_path):
    """Process an image through the autoencoder.

    Parameters
    ----------
    img_path : str
        Path to the input image file

    Returns
    -------
    tuple
        Contains:
        - original_img : PIL.Image.Image
        - transformed_img : torch.Tensor
        - latent_vector : torch.Tensor
        - reconstructed_img : numpy.ndarray
    """
    # Chargement et transformation
    original_img = Image.open(img_path).convert("RGB")
    transformed_img = transform_(original_img)
    img_batch = transformed_img.unsqueeze(0).to(device)

    # Encodage et décodage
    with torch.no_grad():
        latent_vector = model.encode(img_batch)
        reconstructed_vector = model.decode(latent_vector)

    # Conversion pour visualisation
    reconstructed_img = reconstructed_vector.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    return original_img, transformed_img, latent_vector, reconstructed_img


# ===== 6. Affichage des résultats =====
def display_results(original_img, reconstructed_img):
    """Display original and reconstructed images side by side.

    Parameters
    ----------
    original_img : PIL.Image.Image
        Original input image
    reconstructed_img : numpy.ndarray
        Reconstructed output image
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Image reconstruite")
    plt.imshow(reconstructed_img)
    plt.axis("off")

    plt.show()


# ===== 7. Exemple d'utilisation =====
if __name__ == "__main__":
    image_path = "Data bases/Celeb A/Images/img_align_celeba/200003.jpg"

    original, transformed, latent, reconstructed = process_image(image_path)
    print(f"Dimension du vecteur latent: {latent.shape}")

    display_results(original, reconstructed)
