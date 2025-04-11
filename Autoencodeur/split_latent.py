import torch
from utils_autoencoder import load_best_hyperparameters, Autoencoder, device, search_file, transform_
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import os
from PIL import Image

from explore_latent import create_random_vectors, create_black_and_white_vectors, interpolate_vectors, explore_one_coord

def decode_latent_vectors(model, latent_vectors, device):
    """
    Décode une liste vecteurs latents flattened.
    Retourne les images reconstruites np array au format [H, W, C].

    Paramètres
    ----------
    model : class Autoencoder
        Objet de la classe Autoencodeur du fichier utils_autoencoder. 
        Correspond au modèle d'auto-encodage pytorch utilisé pour 
        encoder et décoder les images.
    latent_vectors : np.array
        Tableau contenant un vecteur 1D à reconstruire par ligne.
    device : torch.device
        Device à utiliser pour les fonctions de l'autoencodeur (cuda ou cpu).

    Retours
    -------
    latent_numpy_decoded : no.array
        Images reconstruite sous forme d'un tableau 4D au format (batch, hauteur, largeur, canal)

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

    Paramètres
    ----------
    images_numpy : no.array
        Images sous forme d'un tableau 4D au format (batch, hauteur, largeur, canal)
    nb_cols : int
        Nombre de colonne à utiliser dans la fenêtre d'affichage.
        La valeur par défault est 5.

    Retours
    -------
    None 
        Affiche directement les images côtes à côtes
        dans une fenêtre Matplotlib.
    
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
    Charge deux images dont le nom est donné en paramètre
    à partir du dossier contenant la base de données CelebA.
    Encode ensuite ces images dans l'espace latent d'un autoencodeur.

    Paramètres
    ----------
    model : class Autoencoder
        Objet de la classe Autoencodeur du fichier utils_autoencoder. 
        Correspond au modèle d'auto-encodage pytorch utilisé pour 
        encoder et décoder les images. 
    image1 : str
        Nom de la première image (numéro compris entre 200001 et 202499.jpg)
    image2 : str
        Nom de la deuxième image (numéro compris entre 200001 et 202499.jpg)

    Retours 
    -------
    tuple de np.array
        Couple de tableaux 1D (flat) correspondant aux deux images une fois encodées.

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

def plot_histogram_distance(distances, nb_images=2):
    """
    A partir d'un tableau donnant la distance entre
    les coordonnées de deux (ou la moyenne de plusieurs) vecteurs, 
    affiche l'histogramme des valeurs de distances.

    Permet d'étudier les distances moyennes coordonnées à coordonnées
    entre les vecteurs de l'espace latent, notamment pour trouver
    la taille de mutation optimale dans l'algorithme génétique.

    Paramètres
    ----------
    distances : np.array
        Tableau regroupant la moyenne des distances coordonnées
        à coordonnées entre des vecteurs de l'espace latent.
    nb_images : int
        Nombre d'image utilisées pour calcul le tableau de distances.
        La valeur par défault est de 2.

    Retours
    -------
    None 
        Affiche directement l'histogramme des distances coordonnées à coordonnées
        sur une fenêtre Matplotlib.

    """
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Histogramme des distances coordonnée à coordonnée \n moyennées sur {nb_images} images')
    plt.xlabel('Distance')
    plt.ylabel('Fréquence')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def load_model():
    """
    Charge l'autoencodeur permettant l'encodage et le décodage d'images.

    Paramètres
    ----------
    None
        Utilise directement les paramètres sauvegardés après l'entrainement
        du modèle de l'autoencodeur.

    Retours
    -------
    model : class Autoencoder
        Objet de la classe Autoencodeur du fichier utils_autoencoder. 
        Correspond au modèle d'auto-encodage pytorch utilisé pour 
        encoder et décoder les images. 

    """
    best_params = load_best_hyperparameters("Autoencodeur/best_hyperparameters.pth") # hyperparamètres de l'autoencodeur
    model_filename, _ = search_file("Autoencodeur/conv_autoencoder", extension=".pth", ask_reuse=False)

    model = Autoencoder(nb_channels=best_params['nb_channels'], nb_layers=best_params['nb_layers']).to(device)

    model.load_state_dict(torch.load(model_filename, map_location=device)) # chargement du modèle
    print(f"\n Modèle chargé depuis : {model_filename}")

    model.eval()
    return model

def compute_mean_distance(nb_image_tested):
    """
    Calcul la distance moyenne coordonnées à coordonnées
    entre les vecteurs 1D des images de la base de données encodées 
    dans l'espace latent. 

    Paramètres
    ----------
    nb_image_tested : int
        Nombre de couple d'image à comparer.
    
    Retours 
    -------
    None
        appelle la fonction plot_histogram_distance
        pour afficher l'histogramme des valeurs moyennes de distances
        entre les coordonnées des vecteurs .

    """
    model = load_model()
    arr_dist = np.zeros((nb_image_tested, np.prod(np.array(model.final_shape))//2))
    first_index = 200001
    for i in range(first_index, first_index+nb_image_tested):
        img1, img2 = load_two_images(model, str(i)+".jpg", str(i+1)+".jpg")

        arr_dist[i-first_index] = np.abs(img1 - img2)

    mean_dist = np.mean(arr_dist, axis=0)
    partial_dist = mean_dist[mean_dist <= 2.0]
    plot_histogram_distance(partial_dist, nb_images=2*nb_image_tested)

def main(test_method):
    """
    Fonction principal permettant de lancer les différents tests
    d'exploration des vecteurs de l'espace latent. 
    Se référer à explore_latent.py pour une description de ces tests.

    Paramètres 
    ----------
    test_method : str
        Méthode d'exploration à essayer.

    Retours 
    -------
    None    
        Appelle la fonction display_reconstructed_images pour 
        afficher les images reconstruites à partir des vecteurs
        générés dans l'espace latent.

    """

    model = load_model()

    size = np.prod(np.array(model.final_shape))//2

    try :
        test_method = str(test_method).lower()
        if test_method == "random" :
            latent_vectors = create_random_vectors(size, 30)

        elif test_method == "b&w" :
            latent_vectors = create_black_and_white_vectors(size, 30)

        elif test_method == "interpolation" :
            img1, img2 = 200000 + np.random.randint(1, 2500, 2)
            start_vector, end_vector = load_two_images(model, str(img1)+".jpg", str(img2)+".jpg")
            latent_vectors = interpolate_vectors(start_vector, end_vector, 30)
            # plot_histogram_distance(distances = np.abs(start_vector - end_vector), nb_images = 1) # distance coord by coord)

        elif test_method == "coord":
            latent_vectors = explore_one_coord(np.arange(30, 70), 0.25, size, 30)

        else :
            raise ValueError("Unknown method, please choose 'random', 'b&w', 'interpolation' or 'coord'.")
        
    except  ValueError as e:
        print(f"Erreur : {e}")
        return
    
    reconstructed_latent_vectors = decode_latent_vectors(model, latent_vectors, device)
    display_reconstructed_images(reconstructed_latent_vectors, nb_cols=5)

if __name__ == "__main__":

    #### Test d'exploration ou de distance entre les vecteurs latents ####
    try:
        str_user = input("Que voulez vous faire ? (test distance ou exploration) : ").lower().strip()

        if str_user == "test distance":
            compute_mean_distance(2000)
        elif str_user == "exploration":
            main("interpolation")
        else:
            raise ValueError("Unknown method, please choose 'test distance' or 'exploration'")
    except ValueError as e:
        print(f"Erreur : {e}")