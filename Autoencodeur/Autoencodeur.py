import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn


# ===== 1. Création d'un Dataset personnalisé =====
class CelebADataset(Dataset):
    def __init__(self, folder, transform=None, max_images=None):
        """
        Crée un dataset personnalisé pour charger et prétraiter des images.

        Parameters
        ----------
        folder : str
            Chemin vers le dossier contenant toutes les images.
        transform : torchvision.transforms.Compose, optional
            Transformations à appliquer sur chaque image (redimensionnement, recadrage, conversion en tenseur, etc.).
        max_images : int, optional
            Nombre maximum d'images à utiliser. Si None, toutes les images du dossier sont utilisées.

        Attributes
        ----------
        folder : str
            Chemin du dossier contenant les images.
        transform : torchvision.transforms.Compose or None
            Transformations appliquées sur les images.
        image_files : list of str
            Liste des chemins des fichiers image, triés et potentiellement limités à `max_images`.

        """
        self.folder = folder
        self.transform = transform
        # Lister les fichiers image (extension .jpg, .png, .jpeg)
        self.image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Si max_images est défini, on limite la liste aux premiers 'max_images' fichiers
        self.image_files = self.image_files[:max_images]

    def __len__(self):
        """
        Retourne le nombre total d'images dans le dataset.

        Returns
        -------
        int
            Nombre d'images disponibles dans le dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Récupère une image à l'indice spécifié.

        Parameters
        ----------
        index : int
            Index de l'image à récupérer.

        Returns
        -------
        torch.Tensor
            Image transformée sous forme de tenseur.
        int
            Label fictif (0), utilisé uniquement pour la compatibilité avec PyTorch.
        """
        # Récupère le chemin de l'image à l'indice donné
        image_path = self.image_files[index]

        # Ouvre l'image et convertit en format RGB pour s'assurer d'avoir trois canaux de couleur
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Pour un autoencodeur, la cible est la même image en entrée.
        # Ici, on ne nécessite pas de label réel, donc on retourne un label fictif (0)
        return image, 0


# ===== 2. Définir les transformations =====
image_size = (128, 128)
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # Convertit l'image en tenseur avec des valeurs dans [0,1]
])

# ===== 3. Charger le dataset =====
data_dir = "Data bases/Celeb A/Images/img_align_celeba"  # <-- à modifier en fonction de l'environnement local
dataset = CelebADataset(folder=data_dir, transform=transform_, max_images=2000)

print(f"Nombre total d'images utilisées : {len(dataset)}")

# ===== 4. Split du dataset en 90% train et 10% test =====
total_size = len(dataset)
train_size = int(0.9 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batchSize = 32
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

print(f"Dataset d'entraînement : {train_size} images")
print(f"Dataset de test : {test_size} images")


# ===== 5. Définition de l'Autoencodeur Convolutionnel =====
class ConvAutoencoder(nn.Module):
    """
    Un autoencodeur convolutionnel pour la reconstruction d'images.

    L'autoencodeur est constitué de deux parties :
    - Un encodeur basé sur des couches convolutives qui réduit progressivement la dimension spatiale de l'image
      tout en apprenant une représentation latente.
    - Un décodeur basé sur des couches de convolution transposée qui reconstruit l'image originale à partir
      de la représentation latente.

    Attributs
    ----------
    encoder : nn.Sequential
        Séquence de couches convolutives qui encode l'image d'entrée en un espace latent.
    decoder : nn.Sequential
        Séquence de couches de convolution transposée qui reconstruit l'image à partir de l'espace latent.

    Méthodes
    ---------
    forward(x)
        Applique l'encodeur et le décodeur pour obtenir une image reconstruite.
    """

    def __init__(self):
        """
        Initialise l'architecture de l'autoencodeur.

        L'encodeur réduit progressivement la taille de l'image en appliquant des convolutions avec un stride de 2.
        Le décodeur reconstruit l'image originale en utilisant des convolutions transposées.

        L'image d'entrée est supposée être de taille [3, 128, 128] (RVB).
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
            nn.Sigmoid()  # Normalisation entre [0,1]
        )

    def forward(self, x):
        """
        Effectue un passage avant (forward) à travers l'autoencodeur.

        Parameters
        ----------
        x : torch.Tensor
            Tenseur d'entrée représentant une image de taille [batch_size, 3, 128, 128].

        Returns
        -------
        torch.Tensor
            Image reconstruite de même taille que l'entrée ([batch_size, 3, 128, 128]).
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
