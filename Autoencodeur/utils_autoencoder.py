import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import optuna
from tqdm import tqdm


# ===== Configuration Globale =====

"""
Applique des transformations uniformes sur les images: Redimensionner en 128*128 et normaliser les valeurs
Les réseaux de neurones sont plus stables et efficaces lorsque les entrées sont normalisées
Cela facilite la propagation du gradient et empêche les valeurs trop grandes de ralentir l'entraînement
"""
image_size = (128, 128)
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])

# Load dataset
data_dir = "img_align_celeba"

# ===== For Model Initialization, Loss Function, and Optimizer =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_hyperparameters(path):
    """
    Charge les meilleurs hyperparamètres à partir du fichier sauvegardé à la fin de l’optimisation Optuna. 
    Dictionnaire avec les meilleurs hyperparamètres optimaux testés automatiquement (lr, batch_size, nb_channels, nb_layers...).

    Paramètres
    ----------
    path : str
        Chemin vers le fichier `.pth` contenant les hyperparamètres.

    Retour
    -------
    dict :
        Dictionnaire contenant les hyperparamètres optimaux.
    """
    best_params = torch.load(path)
    print("Hyperparamètres chargés :", best_params)
    return best_params
    
def search_file(base_name, extension = ".pth", ask_reuse = True):
    """
    Gère la création ou la réutilisation d'un fichier existant avec incrémentation automatique.

    Parameters
    ----------
    base_name : str
        Nom de base du fichier (ex: 'conv_autoencoder' ou 'face_autoencoder')
    extension : str, optional
        Extension du fichier (ex: '.pth', '.db'), par default '.pth'
    prompt_label : str, optional
        Nom affiché à l'utilisateur dans la question (ex: "modèle", "base de données")

    Returns
    -------
    filename : str
        Nom du fichier à utiliser
    reuse : bool
        True si on réutilise un fichier existant, False si on crée un nouveau
    """

    reuse_file = False
    file_id = 1
    while os.path.exists(f"{base_name}{'' if file_id == 1 else file_id}{extension}"):
        file_id += 1

    if file_id >1:
        last_id = file_id - 1
        last_filename = f"{base_name}{'' if last_id == 1 else last_id}{extension}"
        if ask_reuse: 
            choice = input(
                f"\nLe fichier '{last_filename}' existe déjà. Voulez-vous :\n"
                f"1. Réutiliser \n"
                f"2. Enregistrer un nouveau\n> "
            )
            if choice == '1':
                reuse_file=True
                return last_filename, reuse_file
            else:
                return f"{base_name}{file_id}{extension}", reuse_file
        else:
            return last_filename, reuse_file
    else:
        f"{base_name}{extension}", reuse_file
              
# ===== Custom Dataset Creation =====
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
        self.image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        self.image_files = self.image_files[
                           :max_images]  # If max_images is set, the list is limited to the first 'max_images' files

    def __len__(self):
        """
        Retourne le nombre total d'images dans le dataset.

        Retour
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

        Paramètres
        ----------
        index : int
            Index de l'image à récupérer.

        Retour
        -------
        torch.Tensor
            L'image transformée en tenseur.
        int
            Dummy label (0), utilisé seulement pour la compatibilité avec les datasets PyTorch
        """

        image_path = self.image_files[index]

        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel color format
        if self.transform:
            image = self.transform(image)

        return image, 0  # The label is not relevant for autoencoders


# ===== Define the Convolutional Autoencoder =====
class Autoencoder(nn.Module):
    """
    Autoencodeur convolutionnel dynamique pour la reconstruction d'images. Le nombre de couches s'adapte aux paramètres envoyés par Optuna. 

    L'autoencodeur a 3 grandes parties principales :

    Encodage : Réduction de la Dimensionnalité
    Un bottleneck (couche dense)
    Décodage : Reconstruction de l’Image

    Les deux sont des séquences de couches convolutives (nn.Sequential).
    L'avantage des séquences est de ne avoir besoin de les définir individuellement dans la fonction forward

    ConvAutoencoder hérite de nn.Module et bénéficie de toutes ses fonctionnalités.
    La classe nn.Module est la classe de base pour tous les modèles de réseaux de neurones PyTorch
    """

    def __init__(self, nb_channels, nb_layers, latent_dim=1024, input_size=128):
        """
        Initialise l'architecture de l'autoencodeur, qui s'adapte en nombre de couches et en capacité de compression (latent_dim)

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

        Bottleneck :
        Après l'encodage, on finit avec un tenseur de forme (C, H, W).
        Il est aplati (.flatten) puis compressé par une couche linéaire de sortie latent_dim défini en argument.
        Cette couche dense forme un goulot d'étranglement et force le réseau à encoder dans un espace vectoriel de dimension 1024.

        Ce vecteur latent est redécompressé par une autre couche dense vers sa forme initiale (.unflatten), 
        avant de passer dans le décodeur.

        Décodeur :
        L'image est reconstruite en inversant l'encodage avec des Convolutions Transposées avec des activations ReLU
        L'autoencodeur prend en entrée une image avec des pixels dans l’intervalle 0,1 (grâce à transforms.ToTensor())
        À la fin du décodeur, on doit reconstruire une image valide, c'est-à-dire avec des pixels entre 0 et 1
        Les convolutions génèrent des valeurs qui peuvent dépasser 1 (ReLU ne borne pas les valeurs positives)

        On veut une sortie entre 0 et 1 pour rester cohérent avec l’entrée.
        Cette cohérence est importante pour l'entraînement, car elle permet de calculer la différence
        entre l'entrée / sortie. Les fonctions de perte supposent que les valeurs E/S sont entre 0 et 1.

        Sigmoid() assure que chaque pixel a une valeur entre 0 et 1.

        Paramètres
        ----------
        nb_channels : int
            Nombre de canaux de sortie pour la dernière couche de l'encodeur. 
        nb_layers : int
            Nombre de couches convolutives dans l’encodeur. 

        latent_dim : int, optionnel (par défaut = 1024)
            Taille du vecteur latent après aplatissement et passage dans la couche dense.
        
        input_size : int, optionnel (par défaut = 128)
            Hauteur/largeur des images d’entrée.
        """

        super().__init__()

        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encodeur convolutionnel
        channels = self.output_channels(nb_channels, nb_layers)
        self.encoder_conv = self.build_layers(channels)

        # Taille finale après convolutions
        self.final_shape, flattened_dim = self.latent_size(self.input_size, self.nb_layers, self.nb_channels)

        # Couche dense
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(flattened_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flattened_dim)
        self.unflatten = nn.Unflatten(1, self.final_shape)

        # Décodeur convolutionnel
        channels_reversed = channels[::-1]
        self.decoder_conv = self.build_layers(channels_reversed, transpose=True)

    def output_channels(self, nb_channels, nb_layers):
        """
        Calcule dynamiquement le nombre de canaux à utiliser dans les couches.

        Construit une liste de canaux allant de 3 (RGB) à `nb_channels`. Elle répartit équitablement 
        l’augmentation des canaux sur `nb_layers` couches.

        On utilise une progression géométrique de raison common_ratio pour interpoler les valeurs intermédiaires. 
        La 1ère couche passe de 3 → 16 (fixé). Puis la dernière couche atteint `nb_channels`, optimisé par Optuna. 

        Le common_ratio ratio est calculé de façon à ce que :  
            16 × ratio^(nb_layers - 1) = nb_channels. nb_layers-1 car la première couche est fixé et ne suit pas ce facteur multiplicatif.
        Evite les sauts brutaux entre les couches.

        Paramètres
        ----------
        nb_channels : int
            Nombre de canaux de sortie à la dernière couche de l’encodeur.
        
        nb_layers : int
            Nombre total de couches dans l’encodeur.

        Retour
        -------
        channels : list[int]
            Liste des dimensions de canaux de chaque couche, de l’entrée (3 canaux) à la sortie (`nb_channels`).
    """
        channels = [3, 16]
        common_ratio = (nb_channels / 16) ** (1 / (nb_layers - 1))
        for i in range(1, nb_layers):
            next_channel = int(round(16 * (common_ratio ** i)))
            channels.append(next_channel)
        return channels

    def build_layers(self, channels, transpose=False):
        """
        Construit une séquence de couches selon l'architecture.

        Itère sur chaque paire successive de canaux dans la liste `channels`
        et ajoute chaque couche une par une.
        Chaque couche est suivie d’une activation ReLU sauf à la fin du décodeur où on applique un Sigmoid.

        kernel_size, stride et padding permettent de diviser par 2 la taille spatiale à chaque couche de l'encodeur

        Paramètres
        ----------
        channels : list[int]
            Liste ordonnée des dimensions de canaux à travers les couches.

        transpose : bool, par défaut False
            Si True, construit un décodeur (ConvTranspose2d).
            Sinon, construit un encodeur (Conv2d).

        Retour
        -------
        nn.Sequential
            Module PyTorch contient la séquence des couches avec activations.
        """
        layers = []
        for i in range(1, len(channels)):
            input_channel = channels[i - 1]
            output_channel = channels[i]
            if transpose:
                layers.append(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, output_padding=1))
                if i != len(channels) - 1:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
        
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
           Image reconstruite à la même taille que l'input.
        """

        x = self.encoder_conv(x)
        x = self.flatten(x)
        latent = self.fc_enc(x)
        if return_latent:
            return latent
        x = self.fc_dec(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder_conv(x)
        return reconstructed

    def encode(self, x):
        """
        Extrait le vecteur latent d'une image.

        Paramètres
        ----------
        x : torch.Tensor
            tenseur qui représente une image [batch_size, 3, 128, 128]

        Retour
        -------
        torch.Tensor
            Vecteur latent de taille [batch_size, 128, 8, 8]
        """

        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.fc_enc(x)
        return x

    def decode(self, z):
        """
        Reconstruit l'image à partir du vecteur latent

        Parameters
        ----------
        z : torch.Tensor
            Vecteur latent de dimension [batch_size, 128, 8, 8].

        Returns
        -------
        torch.Tensor
            Reconstruit l'image de dimension [batch_size, 3, 128, 128]
        """

        x = self.fc_dec(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

    def latent_size(self, input_size, nb_layers, nb_channels):
        """
        Calcule la taille de l'espace latent après nb_layers couches de convolutions
        avec stride=2, padding=1, kernel_size=3.
        """
        if isinstance(input_size, int):
            H = W = input_size
        else:
            H, W = input_size

        H_out = H // (2 ** nb_layers)
        W_out = W // (2 ** nb_layers)

        shape = (nb_channels, H_out, W_out)
        size = nb_channels * H_out * W_out
        return shape, size
    
    def latent_dim_size(self):
        '''
        Retourne la taille du vecteur latent (après la couche dense d'encodage).
        '''
        return self.latent_dim
    
def train_model(data_model, optuna_args=None):
    """
    Entraîne le modèle d'autoencodeur sur un ensemble de données, avec ou sans intégration à Optuna pour l’optimisation
    des hyperparamètres. Intègre aussi un early stopping pour gagner en temps d'entrainement.
   
    Effectue une boucle d'entraînement pendant un nombre d'epochs défini dans data_model. 
    La boucle suit l'évolution des loss d'entrainement et de validation à chaque epoch.

    Si `optuna_args`, chaque epoch est reportée au système d’optimisation d’Optuna qui peut 
    abandonner automatiquement le trial si les performances sont pas bonnes. 

    Sinon la fonction renvoie les pertes d'entraînement et de test finales.

    Paramètres
    ----------
    data_model : dict
        Un dictionnaire qui contient
        'model',  le modèle à entraîner
        'criterion', la fonction de perte
        'optimizer', l’optimiseur (ex: RMSProp, Adam)
        'train_loader', DataLoader pour l’entraînement
        'test_loader', DataLoader pour l’évaluation
        'num_epochs', nombre d’epochs à exécuter

    optuna_args : dict, optionnel
        Contient les arguments liés à Optuna si appelé
        'trial', objet Trial d’Optuna
        'trial_id', nb du trial
        'show_progress', booléen pour afficher la barre de progression par la librairie tqdm

    Retour
    -------
    Si Optuna est utilisé :
        float : la loss finale de test, transmise à Optuna

    Sinon :
        tuple(float, float) :
            - epoch_loss : perte moyenne sur l’ensemble d'entraînement à la fin
            - test_loss : perte moyenne sur l’ensemble de test
    """
    data_model['criterion']
    is_optuna = optuna_args is not None

    train_losses = []
    test_losses = []

    #  Initialisation pour early stopping
    early_stop_patience = 5
    min_delta = 0.0003
    no_improve_count = 0
    best_test_loss = float('inf')

    for epoch in range(data_model['num_epochs']):
        data_model['model'].train()

        if is_optuna:
            desc = f"Trial {optuna_args['trial_id']} | Epoch {epoch+1}" if optuna_args['trial_id'] else f"Epoch {epoch+1}"
            pbar = tqdm(total=len(data_model['train_loader']), desc=desc, leave=False)
        else:
            pbar = None

        running_loss = 0.0
        for images, _ in data_model['train_loader']:
            images = images.to(device)
            data_model['optimizer'].zero_grad()
            outputs = data_model['model'](images)
            loss = data_model['criterion'](outputs, images)
            loss.backward()
            data_model['optimizer'].step()
            running_loss += loss.item() * images.size(0)
            if pbar: 
                pbar.update(1)

         # Compute average loss for the epoch
        epoch_loss = running_loss / len(data_model['train_loader'].dataset)
        train_losses.append(epoch_loss)
            
        if pbar: pbar.close()

        data_model['model'].eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in data_model['test_loader']:
                images = images.to(device)
                outputs = data_model['model'](images)
                loss = data_model['criterion'](outputs, images)
                test_loss += loss.item() * images.size(0)

        test_loss = test_loss / len(data_model['test_loader'].dataset)
        test_losses.append(test_loss)
        print(f"\n Epoch {epoch+1} ; Epoch Loss: {epoch_loss:.4f} ; Test loss: {test_loss:.4f}")

        # Pruning Optuna
        if is_optuna:
            optuna_args['trial'].report(test_loss, epoch)
            if optuna_args['trial'].should_prune():
                print(f"Trial {optuna_args['trial_id']} cancelled at epoch {epoch}. Test loss too bad")
                raise optuna.TrialPruned()
        
        # Early stopping personnalisé
        if best_test_loss - test_loss > min_delta:
            best_test_loss = test_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f" Early stopping déclenché au bout de {epoch+1} epochs")
            break  # ou raise optuna.TrialPruned() si tu veux aussi dire à Optuna que ce trial est nul
    
    if is_optuna:
        return test_loss
        
    else:
        return train_losses, test_losses

def split_data(dataset, train_ratio=0.9):
    """
    Split un dataset en train/test selon le ratio donné en argument.

    Args:
        dataset (torch.utils.data.Dataset): Le dataset complet.
        train_ratio (float): Proportion de données pour l'entraînement (par défaut 0.9).

    Returns:
        tuple: (train_set, test_set, train_size, test_size)
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size


    train_set, test_set = random_split(dataset, [train_size, test_size])

    return train_set, test_set

def latent_size(input_size, nb_layers, nb_channels):
    """
    Calcule la taille de l'espace latent après nb_layers de convolutions
    avec stride=2, padding=1, kernel_size=3.

    Paramètres:
        input_size (int or tuple): taille d'entrée (H, W) ou un seul int
        nb_layers (int): nombre de couches convolutionnelles
        nb_channels (int): nombre de canaux de sortie de la dernière couche

    Retour:
        latent_shape (tuple): (nb_channels, H_out, W_out)
        latent_size (int): nombre total d'éléments dans l'espace latent
    """
    if isinstance(input_size, int):
        H = W = input_size
    else:
        H, W = input_size

    H_out = H // (2 ** nb_layers)
    W_out = W // (2 ** nb_layers)

    shape = (nb_channels, H_out, W_out)
    size = nb_channels * H_out * W_out
    return shape, size
