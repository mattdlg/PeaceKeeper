import torch
from utils_autoencoder import load_best_hyperparameters, Autoencoder, device, search_file
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

def decode_latent_vectors(model, latent_vectors, device):
    """
    Décode une liste vecteurs latents flattened.
    Retourne les images reconstruites np array au format [H, W, C]
    """
    model.eval()
    latent_tensor = torch.tensor(latent_vectors, dtype=torch.float32).to(device)

    with torch.no_grad():
        latent_numpy_decoded = model.decode(latent_tensor).cpu().numpy()

    latent_numpy_decoded = latent_numpy_decoded.transpose(0, 2, 3, 1)  # [B, H, W, C]

    return latent_numpy_decoded


def display_reconstructed_images(images_numpy, nb_cols=5):
    """
    Affiche une grille d'images reconstruites (format [batch, H, W, C])
    """
    nb_images = images_numpy.shape[0]
    nb_rows = ceil(nb_images/nb_cols)

    plt.figure(figsize=(nb_cols * 2, nb_rows * 2))
    for i in range(nb_images):
        plt.subplot(nb_rows, nb_cols, i + 1)
        plt.imshow(images_numpy[i])
        plt.axis("off")
        plt.title(f"Image {i+1}")
    plt.tight_layout()
    plt.show()

def main():

    best_params = load_best_hyperparameters("best_hyperparameters.pth")
    model_filename, _ = search_file("conv_autoencoder", extension=".pth", ask_reuse=False)

    model = Autoencoder(nb_channels=best_params['nb_channels'], nb_layers=best_params['nb_layers']).to(device)

    model.load_state_dict(torch.load(model_filename, map_location=device))
    print(f"\n Modèle chargé depuis : {model_filename}")

    model.eval()


    #ATTENTION A REMPLACER PAR TES VALEURS
    latent_vectors = np.random.randn(15, 1024)

    reconstructed_latent_vectors = decode_latent_vectors(model, latent_vectors, device)
    display_reconstructed_images(reconstructed_latent_vectors, nb_cols=5)

if __name__ == "__main__":
    main()