import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms


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

