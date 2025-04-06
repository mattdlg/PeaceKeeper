import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from utils_autoencoder import (
    CelebADataset, Autoencoder, load_best_hyperparameters, split_data, train_model,
    data_dir, transform_, search_file, device
)

max_images = 50000
train_ratio = 0.7
external_image_path = "img_align_celeba/200001.jpg"

def give_param_model(model, best_params, train_dataset, test_dataset):
    """
    Prépare les paramètres pour l'entraînement du modèle.

    Crée les DataLoaders (entraînement et test) à partir des datasets et configure la fonction de perte (MSE) et l’optimiseur (RMSProp).

    `MSELoss` et non 'l1' a été choisi car des tests de validations préliminaires Optuna ont montré que c'était la meilleure
    Pareil pour RMSProp qui permet d’ajuster dynamiquement le learning rate par paramètre.
    Idem pour batch_size.
    Le test_loader est en shuffle=True pour diversifier les images reconstruites à chaque test.

    Paramètres
    ----------
    model : nn.Module
        L'autoencodeur à entraîner.
    best_params : dict
        Dictionnaire d’hyperparamètres optimisés.
    train_dataset : Dataset
        Sous-ensemble de données d'entraînement.
    test_dataset : Dataset
        Sous-ensemble de données de test.

    Retour
    -------
    dict :
        Un dictionnaire structuré avec tous les composants nécessaires à l’appel de train_model().
    """
    return {
        "model": model,
        "criterion": nn.MSELoss(),
        "optimizer": optim.RMSprop(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']),
        "train_loader": DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True),
        "test_loader": DataLoader(test_dataset, batch_size=1, shuffle=True),
        "num_epochs": 20,
    }

def test_model_on_image(model, device, image_path):
    """
    Teste le modèle entraîné sur une image externe qui ne fait pas parti du dataset d'entraînement.

    Permet de reconstruire l'image à partir de l'original qu'il n'a jamais vu.
    qu’il n’a jamais vue. Elle est transformée pour correspondre aux dimensions 128x128 du dataset
    Elle est encodée en vecteur latent puis décodée.
    L’évaluation se fait sans gradients (torch.no_grad) pour économiser mémoire et temps.

    Paramètres
    ----------
    model : nn.Module
        Autoencodeur entraîné.
    device : torch.device
        CPU ou GPU utilisé.
    image_path : str
        Chemin vers l’image à tester.

    Retour
    -------
    original_img : PIL.Image
        Image d’origine (non transformée).
    reconstructed_img : numpy.ndarray
        Image reconstruite par le modèle, au format compatible avec matplotlib (H, W, C).
    """
    external_img = Image.open(image_path).convert("RGB")
    external_img_transformed = transform_(external_img)
    external_img_batch = external_img_transformed.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        latent_vector = model.encode(external_img_batch)
        print("Latent vector shape:", model.final_shape)
        reconstructed_batch = model.decode(latent_vector)

    reconstructed_img_tensor = reconstructed_batch.squeeze(0).cpu()
    reconstructed_img = reconstructed_img_tensor.numpy().transpose(1, 2, 0)

    return external_img, reconstructed_img

def show_images(original_img, reconstructed_img):
    '''
    Affiche l’image d’origine et l’image reconstruite par le modèle. Visualisation qualitative.

    Paramètres
    ----------
    original_img : PIL.Image
        Image originale chargée depuis le disque.
    reconstructed_img : np.ndarray
        Image reconstruite par le modèle.
    '''
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original external image")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed image")
    plt.imshow(reconstructed_img)
    plt.axis("off")
    plt.show()

def main():
    # ===== 1. Charger les hyperparamètres optimisés ====
    best_params = load_best_hyperparameters("best_hyperparameters.pth")

    # ===== 2. Préparation des données =====
    dataset = CelebADataset(data_dir, transform_, max_images)
    train_dataset, test_dataset =split_data(dataset, train_ratio)

    model_filename, reuse_model = search_file("conv_autoencoder", extension=".pth")
    model = Autoencoder(nb_channels=best_params['nb_channels'], nb_layers=best_params['nb_layers']).to(device)

    # ===== 3. Entraînement ou chargement =====
    if reuse_model:
        model.load_state_dict(torch.load(model_filename, map_location=device))
        print(f"\n Modèle chargé depuis : {model_filename}")
    
    else: 
        data_model = give_param_model(model, best_params, train_dataset, test_dataset)
        epoch_loss, test_loss = train_model(data_model)
        torch.save(model.state_dict(), model_filename)
        print(f"\nModèle sauvegardé sous : {model_filename}")

    # ===== 6. Test sur une image externe =====
    original_img, reconstructed_img = test_model_on_image(model, device, external_image_path)

    # ===== 7. Visualisation =====
    show_images(original_img, reconstructed_img)
    
if __name__ == "__main__":
    main()


