import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import optuna
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ===== Custom Dataset Creation =====
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

        self.image_files = self.image_files[
                           :max_images]  # If max_images is set, the list is limited to the first 'max_images' files

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

        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel color format
        if self.transform:
            image = self.transform(image)

        return image, 0  # The label is not relevant for autoencoders

# ===== Define Image Transformations =====
image_size = (128, 128)
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # Converts image to tensor with values in [0,1]
])

# ===== Load Dataset =====
data_dir = "img_align_celeba" # <-- Path à update en fonction de là où sont stockés les img
dataset = CelebADataset(folder=data_dir, transform=transform_, max_images=2000)

# ===== Define the Convolutional Autoencoder =====
class Autoencoder(nn.Module):
    def __init__(self, nb_channels, nb_layers, latent_dim=1024, input_size=128):
        super().__init__()

        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Convolutional Encoder
        channels = self.output_channels(nb_channels, nb_layers)
        self.encoder_conv = self.build_layers(channels)

        # Final shape after convolutions
        self.final_shape, flattened_dim = self.latent_size(self.input_size, self.nb_layers, self.nb_channels)

        # Density layer
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(flattened_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flattened_dim)
        self.unflatten = nn.Unflatten(1, self.final_shape)

        # Convolutional Decoder
        channels_reversed = channels[::-1]
        self.decoder_conv = self.build_layers(channels_reversed, transpose=True)

    def output_channels(self, nb_channels, nb_layers):
        channels = [3, 16]
        common_ratio = (nb_channels / 16) ** (1 / (nb_layers - 1))
        for i in range(1, nb_layers):
            next_channel = int(round(16 * (common_ratio ** i)))
            channels.append(next_channel)
        return channels

    def build_layers(self, channels, transpose=False):
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
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.fc_enc(x)
        return x

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

    def latent_size(self, input_size, nb_layers, nb_channels):
        """
        Calcule la taille de l'espace latent après nb_layers de convolutions
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

    

# ===== For Model Initialization, Loss Function, and Optimizer =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 7. Training Loop =====
def train_model(data_model, optuna_args=None):
    data_model['criterion']
    is_optuna = optuna_args is not None

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
        return epoch_loss, test_loss

def split_data(dataset, train_ratio=0.9):
    """
    Split un dataset en train/test selon un ratio donné.

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

    Args:
        input_size (int or tuple): taille d'entrée (H, W) ou un seul int
        nb_layers (int): nombre de couches convolutionnelles
        nb_channels (int): nombre de canaux de sortie de la dernière couche

    Returns:
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

