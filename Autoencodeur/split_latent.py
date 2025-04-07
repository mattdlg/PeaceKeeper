import torch
from utils_autoencoder import load_best_hyperparameters, Autoencoder, device, search_file, transform_
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import os
from PIL import Image

from explore_latent import create_random_vectors, create_black_and_white_vectors, interpolate_vectors

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

def load_two_images(model, image1, image2):
    """
    """
    filename1 =  os.path.join(".", "Data Bases", "Celeb A", "Images", "selected_images", image1)
    filename2 =  os.path.join(".", "Data Bases", "Celeb A", "Images", "selected_images", image2)

    img1 = Image.open(filename1).convert("RGB")
    img2 = Image.open(filename2).convert("RGB")

    tensor_img1 = transform_(img1).unsqueeze(0).to(device)
    tensor_img2 = transform_(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_img1 = model.encode(tensor_img1).cpu().numpy()
        encoded_img2 = model.encode(tensor_img2).cpu().numpy()

    return encoded_img1[0], encoded_img2[0]

def plot_histogram_distance(distances, nb_images=1):
    """
    """
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Histogramme des distances coordonnée à coordonnée \n moyennées sur {nb_images} images')
    plt.xlabel('Distance')
    plt.ylabel('Fréquence')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def load_model():
    best_params = load_best_hyperparameters("Autoencodeur/best_hyperparameters.pth")
    model_filename, _ = search_file("Autoencodeur/conv_autoencoder", extension=".pth", ask_reuse=False)

    model = Autoencoder(nb_channels=best_params['nb_channels'], nb_layers=best_params['nb_layers']).to(device)

    model.load_state_dict(torch.load(model_filename, map_location=device))
    print(f"\n Modèle chargé depuis : {model_filename}")

    model.eval()
    return model

def compute_mean_distance(nb_image_tested):
    model = load_model()
    arr_dist = np.zeros((nb_image_tested, np.prod(np.array(model.final_shape))//2))
    first_index = 200001
    for i in range(first_index, first_index+nb_image_tested):
        img1, img2 = load_two_images(model, str(i)+".jpg", str(i+1)+".jpg")

        arr_dist[i-first_index] = np.abs(img1 - img2)

    mean_dist = np.mean(arr_dist, axis=0)
    partial_dist = mean_dist[mean_dist <= 2.0]
    plot_histogram_distance(partial_dist, nb_images=nb_image_tested)

def main():

    model = load_model()

    size = np.prod(np.array(model.final_shape))//2

    #ATTENTION A REMPLACER PAR TES VALEURS
    # latent_vectors = np.random.randn(15, size)
    # latent_vectors = create_random_vectors(size, 30)
    # latent_vectors = create_black_and_white_vectors(size, 30)
    start_vector, end_vector = load_two_images(model, "200001.jpg", "200003.jpg")

    plot_histogram_distance(distances = np.abs(start_vector - end_vector), nb_images = 1) # distance coord by coord)

    latent_vectors = interpolate_vectors(start_vector, end_vector, 30)
    
    reconstructed_latent_vectors = decode_latent_vectors(model, latent_vectors, device)
    display_reconstructed_images(reconstructed_latent_vectors, nb_cols=5)

if __name__ == "__main__":
    compute_mean_distance(2000)
    # main()