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
    1.2 (12/04/2025)
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

    Définit une classe de dataset personnalisé pour charger et prétraiter les images de CelebA en vue de leur
     utilisation dans l'autoencodeur.

    Cette classe hérite de torch.utils.data.Dataset (Dataset est un parent de CelebADataset)
     Or, Dataset vient de torch.utils.data, donc CelebADataset hérite indirectement de torch.utils.data.Dataset)

    Comme la classe CelebADataset hérite de Dataset, qui est une classe abstraite,
     elle doit redéfinir __len__ et __getitem__.
     Dans PyTorch, un dataset personnalisé doit obligatoirement implémenter ces deux méthodes
      pour être utilisé correctement par un DataLoader
    """

    def __init__(self, folder, transform=None, max_images=None):
        """
        Création d'une instance de la classe CelebADataset.

        La fonction liste tous les fichiers du dossier spécifié, filtre uniquement les images.
        Les images sont triées, utile pour la reproductibilité de l’entraînement.
         Utile aussi pour le débug en s'assurant que les fichiers sont toujours traités dans le même ordre

        Parameters
        ----------
        folder : str
            Path to the folder containing images.
        transform : torchvision.transforms.Compose, optional
            Transformations applied to each image (resizing, cropping, conversion to tensor, etc.).
        max_images : int, optional
            Maximum number of images to use. If None, all images in the folder are used.
        """
        self.folder = folder
        self.transform = transform

        # Lister et trier tous les fichiers du dossier folder:
        # os.listdir(folder): Retourne une liste des fichiers et dossiers présents dans folder

        # [os.path.join(folder, f) for f in os.listdir(folder)]:
        # Concatène le dossier folder avec chaque nom de fichier pour avoir le chemin complet

        # sorted([...]): Trie les fichiers par ordre alphabétique.

        # if f.lower().endswith(('.jpg', '.jpeg', '.png')) :
        # filtre les fichiers pour ne garder que les images avec une extension .jpg, .jpeg ou .png
        self.image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Si max_images est défini, on garde uniquement les 'max_images' premières images
        self.image_files = self.image_files[:max_images]

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
        Permet de charger une image du dataset à un index donné et de lui appliquer des transformations
         avant de la retourner sous forme de tenseur.

        Cette méthode est obligatoirement exécutée, de manière indirecte, à chaque chargement d'image
         pendant l'entraînement

        L'ajout du label 0 en retour permet de garder la compatibilité avec PyTorch, car en général, les datasets
         sont utilisés pour des problèmes supervisés, où on retourne (entrée, label)
        Si l'autoencodeur n'a pas besoin de labels, certaines fonctions ou outils de PyTorch peuvent s'attendre à un
         format (entrée, cible)

        Parameters
        ----------
        index : int
            Index of the image to retrieve.

        Returns
        -------
        torch.Tensor
            The transformed image as a tensor.
        int
            Dummy label (0), utilisé seulement pour la compatibilité avec les datasets PyTorch
        """
        image_path = self.image_files[index]  # Récupère le chemin du fichier image correspondant à l’index donné

        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel color format
        if self.transform:
            image = self.transform(image)

        return image, 0  # The label is not relevant for autoencoders


# ===== 5. Define the Convolutional Autoencoder =====
class ConvAutoencoder(nn.Module):
    """
    Autoencodeur convolutionnel classique pour la reconstruction d'images

    L'autoencodeur a deux parties principales :

    Encodage : Réduction de la Dimensionnalité
    Décodage : Reconstruction de l’Image

    Les deux sont des séquences de couches convolutives (nn.Sequential).
     L'avantage des séquences est de ne avoir besoin de les définir individuellement dans la fonction forward

    ConvAutoencoder hérite de nn.Module et bénéficie de toutes ses fonctionnalités.
    La classe nn.Module est la classe de base pour tous les modèles de réseaux de neurones PyTorch

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

        L'encodeur :
            L'entrée est une image 3x128x128 (3 canaux pour RGB, 128x128 pixels).

            Chaque couche convolutive réduit la résolution tout en augmentant le nombre de filtres
            Lorsqu’un filtre passe sur une image, il effectue un produit scalaire entre la matrice des pixels
             et la matrice du filtre.
            Cela donne des valeurs positives et négatives en sortie.
            Valeurs positives → Forte activation (la région de l’image correspond bien au filtre).
            Valeurs négatives → Désaccord avec le filtre (la région ne correspond pas au motif recherché).

                Exemple d'interprétation de valeurs après passage d'un filtre :
                Si on cherche un bord clair sur fond sombre, alors les pixels dans la direction du bord donneront
                 une forte activation positive.
                Les pixels dans l'autre sens pourraient donner des valeurs négatives, qui ne sont pas toujours utiles.

            À chaque convolution, on introduit de la non-linéarité avec la fonction ReLU (Rectified Linear Unit) :
            ReLU est définie mathématiquement par : f(x) = max(0,x)
                Cela signifie :
                Si x > 0 : L'entrée est conservée (f(x) = x)
                Si x < 0 : L'entrée est mise à zéro

                Exemple de fonctionnement de ReLU :
                Si on applique ReLU à un ensemble de valeurs : [-3.0, -1.5, 0.0, 1.2, 4.5, -2.7]
                Après ReLU : [0.0, 0.0, 0.0, 1.2, 4.5, 0.0]
            L'activation ReLU simplifie le calcul, accélère l’apprentissage, et évite les problèmes de saturation du
             gradient

            Après cette étape, l'image est représentée par un vecteur latent de 128x8x8


        Décodeur :
        L'image est reconstruite en inversant l'encodage avec des Convolutions Transposées avec des activations ReLU
        L'autoencodeur prend en entrée une image avec des pixels dans l’intervalle 0,1 (grâce à transforms.ToTensor())
        À la fin du décodeur, on doit reconstruire une image valide, c'est-à-dire avec des pixels entre 0 et 1
        Les convolutions génèrent des valeurs qui peuvent dépasser 1 (ReLU ne borne pas les valeurs positives)

        On veut une sortie entre 0 et 1 pour rester cohérent avec l’entrée.
        Cette cohérence est importante pour l'entraînement, car elle permet de calculer la différence
         entre l'entrée / sortie. Les fonctions de perte supposent que les valeurs E/S sont entre 0 et 1.

        Sigmoid() assure que chaque pixel a une valeur entre 0 et 1.
        """
        super(ConvAutoencoder, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 3x128x128 -> 16x64x64
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x64x64 -> 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32x32 -> 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 128x8x8
            nn.ReLU(True),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x8x8 -> 64x16x16
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
        Cette méthode définit comment les données passent à travers l’autoencodeur

        L'image passe à travers l’encodeur jusqu'à obtenir une représentation latente.

        Si return_latent=True, la fonction s’arrête ici et renvoie directement la représentation latente,
         utile si on veut ajouter l'algo génétique

        Si return_latent=False, on continue :
            La représentation latente latent passe dans le décodeur.
            L’image reconstruite est renvoyée, ayant la même forme que l’image d’entrée

        Parameters
        ----------
        x : torch.Tensor
            Un tenseur d’entrée représentant une image de la taille [batch_size, 3, 128, 128] (images RGB 128x128)
        return_latent : bool, optional, par défaut False
            If True: permet de récupérer uniquement la représentation latente au lieu de reconstruire l’image

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
            Input tensor representing an image of shape [batch_size, 3, 128, 128]

        Returns
        -------
        torch.Tensor
            Latent vector of shape [batch_size, 128, 8, 8]
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
            Reconstructed image of shape [batch_size, 3, 128, 128]
        """
        return self.decoder(z)


if __name__ == '__main__':
    # ===== 2. Define Image Transformations =====
    """
    Applique des transformations uniformes sur les images: Redimensionner en 128*128 et normaliser les valeurs
    Les réseaux de neurones sont plus stables et efficaces lorsque les entrées sont normalisées
    Cela facilite la propagation du gradient et empêche les valeurs trop grandes de ralentir l'entraînement
    """
    image_size = (128, 128)
    transform_ = transforms.Compose(
        [  # Garantit que toutes les images ont exactement la même taille, la même normalisation
            transforms.Resize(image_size),  # Redimensionne l'image
            transforms.CenterCrop(image_size),
            # S'assurer que toutes les images ont une taille uniforme sans distorsion
            # Convertir l'image en un tenseur de forme (Canaux, Hauteur, Largeur) avec des valeurs normalisées
            # entre 0 et 1 (au lieu de 0-255) :
            transforms.ToTensor(),
        ])

    # ===== 3. Load Dataset =====
    data_dir = "/Data Bases/Celeb A/Images/img_align_celeba"  # <-- Path à update en fonction de là où sont stockés les img
    dataset = CelebADataset(folder=data_dir, transform=transform_, max_images=2000)

    print(f"Total number of images used : {len(dataset)}")

    # ===== 4. Split Dataset (90% Train, 10% Test) =====
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ===== 4a. Charger les données d'entraînement et de test par mini-batches avec un DataLoader=====
    """
    DataLoader charge des mini-batches d’images au lieu d’une seule.
    
    L'utilisation d'un DataLoader pour charger des images par lots (au lieu de une par une) permet d'utiliser
     plus efficacement la mémoire GPU et de paralléliser le calcul
    
    Les gradients calculés sur un mini-batch sont plus représentatifs que ceux calculés sur une seule image, ce qui
     stabilise l'optimisation
    
    Exemple de chargement de batches:
        Si le dataset d'entrainement contient 1000 images et batch_size = 32 :
        train_loader va créer 1000 / 32 = 31,25 ≈ 31 batches
        Il va traiter 31 batches de 32 images + 1 batch final avec 8 images
        Après un epoch, toutes les images auront été vues une fois 
    """
    batchSize = 32
    # Chaque itération du training récupérera 32 images à la fois, permet de ne pas charger tout le dataset d'un coup

    # Note : Un DataLoader n'est pas une fonction mais une classe.
    # Ils ne sont pas des listes d'images, mais des itérateurs qui retournent des mini-batches
    # de tenseurs (batch_size, 3, 128, 128)
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

    print(f"Dataset d'entraînement : {train_size} images")
    print(f"Dataset de test : {test_size} images")

    # ===== 6. Model Initialization, Loss Function, and Optimizer =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(
        device)  # ConvAutoencoder() crée une instance du modèle, qui est transféré sur le G/CPU

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
        - Avec chaque epoch, il ajuste ses poids en fonction de la backpropagation.
        - Après plusieurs époques, il s'améliore progressivement en réduisant l'erreur de reconstruction.
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
            images = images.to(device)  # On met les images sur le même device que le modèle
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

    # ===== 8. Test on an external image by getting first latent vector then reconstruction =====
    # Specify the path to the external image (ensure it is not in the training or test set)
    external_image_path = "/content/img_align_celeba/200001.jpg"  # <-- Modify if necessary

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
    reconstructed_img_tensor = reconstructed_batch.squeeze(0).cpu()  # Convert from [C, H, W] to [H, W, C]
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
