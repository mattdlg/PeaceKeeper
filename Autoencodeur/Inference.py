import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ===== 1. Définition de l'architecture (identique à celle utilisée lors de l'entraînement) =====
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128x128 -> 16x64x64
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x64x64 -> 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32x32 -> 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x16x16 -> 128x8x8
            nn.ReLU(True),
        )
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x8x8 -> 64x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x16x16 -> 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x32 -> 16x64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x64x64 -> 3x128x128
            nn.Sigmoid()  # Pour obtenir des sorties dans [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# ===== 2. Configuration =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 3. Chargement du modèle enregistré =====
model = ConvAutoencoder().to(device)
# Chargez les poids sauvegardés (assurez-vous que 'conv_autoencoder.pth' est dans le même dossier ou modifiez le chemin)
model.load_state_dict(torch.load('conv_autoencoder.pth', map_location=device))
model.eval()  # Met le modèle en mode évaluation

# ===== 4. Définition de la transformation pour les images =====
image_size = (128, 128)
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # Convertit l'image en tenseur avec des valeurs normalisées dans [0,1]
])

# ===== 5. Chargement et transformation d'une image externe =====
external_image_path = "Data bases/Celeb A/Images/img_align_celeba/002001.jpg"  # <-- Modifiez ce chemin avec celui de votre image externe
original_img = Image.open(external_image_path).convert("RGB")
transformed_img = transform_(original_img)

# Ajouter une dimension batch car le modèle attend [batch, channels, height, width]
img_batch = transformed_img.unsqueeze(0).to(device)

# ===== 6. Inférence =====
with torch.no_grad():
    reconstructed_batch = model(img_batch)

# Retirer la dimension batch et convertir le tenseur en image numpy
reconstructed_img_tensor = reconstructed_batch.squeeze(0).cpu()
reconstructed_img = reconstructed_img_tensor.numpy().transpose(1, 2, 0)

# ===== 7. Affichage de l'image originale et de sa reconstruction =====
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
