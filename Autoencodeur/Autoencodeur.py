import os
from PIL import Image
from torch.utils.data import Dataset


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
