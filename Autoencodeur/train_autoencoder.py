import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from utils_autoencoder import CelebADataset, Autoencoder, split_data, train_model

# ===== 1. Charger les hyperparamètres optimisés ===== Donne
best_params = torch.load('best_hyperparameters.pth')
print("Hyperparamètres chargés :", best_params)

image_size = (128, 128)
data_dir = "img_align_celeba"
transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])

# ===== 2. Préparation des données =====
dataset = CelebADataset(folder=data_dir, transform=transform_, max_images=20000)
train_dataset, test_dataset =split_data(dataset, train_ratio=0.7)

# ===== 3. Gestion automatique du nom de fichier =====
base_name = "conv_autoencoder"
model_id = 1

# Chercher les modèles déjà existants
while os.path.exists(f"{base_name}{'' if model_id == 1 else model_id}.pth"):
    model_id += 1

# Proposer d'utiliser le dernier existant (model_id - 1) ou de créer un nouveau
if model_id > 1:
    last_model_id = model_id - 1
    last_filename = f"{base_name}{'' if last_model_id == 1 else last_model_id}.pth"
    choice = input(f"\nLe fichier '{last_filename}' existe déjà. Voulez-vous :\n1. Réutiliser ce modèle\n2. Enregistrer un nouveau modèle\n> ")
    
    if choice == '1':
        model_filename = last_filename
        reutiliser_modele = True
    else:
        model_filename = f"{base_name}{model_id}.pth"
        reutiliser_modele = False
else:
    print("Entrée invalide. Par défaut, un nouveau modèle sera créé.")
    model_filename = f"{base_name}{model_id}.pth"
    reutiliser_modele = False

# Création du modèle avec les bons hyperparamètres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(nb_channels=best_params['nb_channels'], nb_layers=best_params['nb_layers']).to(device)

# ===== 5. Entraînement ou chargement =====
if reutiliser_modele:
    model.load_state_dict(torch.load(model_filename, map_location=device))
    print(f"\n Modèle chargé depuis : {model_filename}")
else:
    data_model = {
        "model": model,
        "criterion": nn.MSELoss(),
        "optimizer": optim.RMSprop(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']),
        "train_loader": DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True),
        "test_loader": DataLoader(test_dataset, batch_size=1, shuffle=True),
        "num_epochs": 20,
    }
    epoch_loss, test_loss = train_model(data_model)
    torch.save(model.state_dict(), model_filename)
    print(f"\nModèle sauvegardé sous : {model_filename}")

# ===== 6. Test sur une image externe =====
external_image_path = "img_align_celeba/200003.jpg"
external_img = Image.open(external_image_path).convert("RGB")
external_img_transformed = transform_(external_img)
external_img_batch = external_img_transformed.unsqueeze(0).to(device)

model.eval()

with torch.no_grad():
    latent_vector = model.encode(external_img_batch)

print("Latent vector shape:", model.final_shape)

with torch.no_grad():
    reconstructed_batch = model.decode(latent_vector)

reconstructed_img_tensor = reconstructed_batch.squeeze(0).cpu()
reconstructed_img = reconstructed_img_tensor.numpy().transpose(1, 2, 0)

# ===== 7. Visualisation =====
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original external image")
plt.imshow(external_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed image")
plt.imshow(reconstructed_img)
plt.axis("off")

plt.show()