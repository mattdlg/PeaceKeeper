# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from PIL import Image
# from torchvision import transforms
#
#
# # ===== 1. Définition de l'architecture mise à jour =====
# class ConvAutoencoder(nn.Module):
#     """Convolutional Autoencoder model for image compression and reconstruction.
#
#     Attributes
#     ----------
#     encoder : torch.nn.Sequential
#         Sequential container of encoder layers
#     decoder : torch.nn.Sequential
#         Sequential container of decoder layers
#     """
#
#     def __init__(self):
#         """Initialize the ConvAutoencoder with encoder and decoder layers."""
#
#         super(ConvAutoencoder, self).__init__()
#
#         # --- Encoder ---
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#         )
#
#         # --- Decoder ---
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, return_latent=False):
#         """Forward pass through the autoencoder.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, 3, 128, 128)
#         return_latent : bool, optional
#             Whether to return the latent vector instead of reconstruction
#
#         Returns
#         -------
#         torch.Tensor
#             Reconstructed image if return_latent=False, latent vector otherwise
#         """
#         latent_vector = self.encoder(x)
#         if return_latent:
#             return latent_vector
#         return self.decoder(latent_vector)
#
#     def encode(self, x):
#         """Encode an input image to latent space.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, 3, 128, 128)
#
#         Returns
#         -------
#         torch.Tensor
#             Latent vector of shape (batch_size, 128, 8, 8)
#         """
#         return self.encoder(x)
#
#     def decode(self, z):
#         """Decode a latent vector to image space.
#
#         Parameters
#         ----------
#         z : torch.Tensor
#             Latent vector of shape (batch_size, 128, 8, 8)
#
#         Returns
#         -------
#         torch.Tensor
#             Reconstructed image of shape (batch_size, 3, 128, 128)
#         """
#         return self.decoder(z)
#
#
# # ===== 2. Configuration =====
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ===== 3. Chargement du modèle =====
# model = ConvAutoencoder().to(device)
# model.load_state_dict(torch.load('conv_autoencoder.pth', map_location=device))
# model.eval()
#
# # ===== 4. Transformation des images =====
# image_size = (128, 128)
# transform_ = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
# ])
#
#
# # ===== 5. Fonction d'inférence complète =====
# def process_image(img_path):
#     """Process an image through the autoencoder.
#
#     Parameters
#     ----------
#     img_path : str
#         Path to the input image file
#
#     Returns
#     -------
#     tuple
#         Contains:
#         - original_img : PIL.Image.Image
#         - transformed_img : torch.Tensor
#         - latent_vector : torch.Tensor
#         - reconstructed_img : numpy.ndarray
#     """
#     # Chargement et transformation
#     original_img = Image.open(img_path).convert("RGB")
#     transformed_img = transform_(original_img)
#     img_batch = transformed_img.unsqueeze(0).to(device)
#
#     # Encodage et décodage
#     with torch.no_grad():
#         latent_vector = model.encode(img_batch)
#         reconstructed_vector = model.decode(latent_vector)
#
#     # Conversion pour visualisation
#     reconstructed_img = reconstructed_vector.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#
#     return original_img, transformed_img, latent_vector, reconstructed_img
#
#
# # ===== 6. Affichage des résultats =====
# def display_results(original_img, reconstructed_img):
#     """Display original and reconstructed images side by side.
#
#     Parameters
#     ----------
#     original_img : PIL.Image.Image
#         Original input image
#     reconstructed_img : numpy.ndarray
#         Reconstructed output image
#     """
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.title("Image originale")
#     plt.imshow(original_img)
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.title("Image reconstruite")
#     plt.imshow(reconstructed_img)
#     plt.axis("off")
#
#     plt.show()
#
#
# # ===== 7. Exemple d'utilisation =====
# if __name__ == "__main__":
#     image_path = "Data bases/Celeb A/Images/img_align_celeba/200003.jpg"
#
#     original, transformed, latent, reconstructed = process_image(image_path)
#     print(f"Dimension du vecteur latent: {latent.shape}")
#
#     display_results(original, reconstructed)

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import random
import os
import torch
import numpy as np
from torchvision import transforms


# ---------- Classe Principale ----------
class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Reconnaissance Faciale - Police Scientifique")
        self.master.geometry("1200x800")
        self.master.configure(bg='#f0f0f0')

        # Configuration initiale
        self.title_font = ('Arial', 14, 'bold')
        self.button_font = ('Arial', 10)
        self.primary_color = '#2c3e50'
        self.secondary_color = '#3498db'

        # Chemin des images
        self.image_folder = "Data bases/Celeb A/Images/img_align_celeba/"
        self.all_images = self.load_image_list()
        self.used_images = set()

        # Initialisation du modèle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.create_transforms()

        # Interface utilisateur
        self.create_widgets()
        self.load_new_images()

    def load_image_list(self):
        """Charge la liste des images valides"""
        return [f for f in os.listdir(self.image_folder)
                if f.endswith('.jpg') and f.split('.')[0].isdigit()
                and int(f.split('.')[0]) >= 200001]

    def create_widgets(self):
        """Crée les éléments de l'interface"""
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Titre
        title_label = ttk.Label(main_frame,
                                text="Veuillez sélectionner une image pour la reconstruction",
                                font=self.title_font,
                                foreground=self.primary_color)
        title_label.pack(pady=10)

        # Frame des images
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=20)

        ttk.Button(control_frame,
                   text="Nouvelles Images",
                   command=self.confirm_new_images).pack(side=tk.LEFT, padx=10)

        ttk.Button(control_frame,
                   text="Quitter",
                   command=self.confirm_exit).pack(side=tk.RIGHT, padx=10)

    # ---------- Gestion du Modèle ----------
    def load_model(self):
        """Charge le modèle entraîné"""

        class ConvAutoencoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    torch.nn.ReLU(True),
                )
                # Decoder
                self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.Sigmoid()
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = ConvAutoencoder().to(self.device)
        model.load_state_dict(torch.load('conv_autoencoder.pth', map_location=self.device))
        model.eval()
        return model

    def create_transforms(self):
        """Crée les transformations d'images"""
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
        ])

    # ---------- Logique Métier ----------
    def load_new_images(self):
        """Charge 6 nouvelles images"""
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        available_images = [img for img in self.all_images if img not in self.used_images]
        selected_images = random.sample(available_images, min(6, len(available_images)))
        self.used_images.update(selected_images)

        for i, img_name in enumerate(selected_images):
            img_path = os.path.join(self.image_folder, img_name)
            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)

                btn = ttk.Button(self.image_frame,
                                 image=photo,
                                 command=lambda p=img_path: self.process_selection(p))
                btn.image = photo
                btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)
            except Exception as e:
                print(f"Erreur chargement image {img_name}: {e}")

    def process_selection(self, img_path):
        """Gère la sélection d'image"""
        if messagebox.askyesno("Confirmation", "L'algorithme sera appliqué sur cette image.\nVoulez-vous continuer ?"):
            self.show_reconstruction(img_path)

    def show_reconstruction(self, img_path):
        """Affiche les résultats de reconstruction"""
        original, reconstructed = self.process_image(img_path)

        self.clear_interface()

        result_frame = ttk.Frame(self.image_frame)
        result_frame.pack(expand=True, pady=20)

        # Affichage des images
        self.display_image(original, result_frame, "left")
        self.display_image(reconstructed, result_frame, "right")

        # Bouton de retour
        ttk.Button(self.image_frame,
                   text="Retour au menu",
                   command=self.load_new_images).pack(pady=20)

    def process_image(self, img_path):
        """Traite une image via l'autoencodeur"""
        img = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(tensor_img).cpu().numpy()[0].transpose(1, 2, 0)

        return img, Image.fromarray((reconstructed * 255).astype(np.uint8))

    # ---------- Utilitaires ----------
    def display_image(self, img, frame, side):
        """Affiche une image dans l'interface"""
        disp_img = ImageTk.PhotoImage(img.resize((400, 400)))
        label = ttk.Label(frame, image=disp_img)
        label.image = disp_img
        label.pack(side=side, padx=20)

    def clear_interface(self):
        """Nettoie l'interface"""
        for widget in self.image_frame.winfo_children():
            widget.destroy()

    def confirm_new_images(self):
        """Confirmation de rechargement"""
        if messagebox.askyesno("Nouvelles Images", "Cela va charger 6 nouvelles images non visionnées.\nContinuer ?"):
            self.load_new_images()

    def confirm_exit(self):
        """Confirmation de sortie"""
        if messagebox.askyesno("Quitter", "Êtes-vous sûr de vouloir quitter l'application ?"):
            self.master.destroy()


# ---------- Lancement de l'application ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
