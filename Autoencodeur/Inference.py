import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import random
import os
import torch
import numpy as np
from AlgoGenetique import algo_genetique_parallel
from torchvision import transforms


# ---------- Classe Principale ----------
class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.attributes("-fullscreen", True) # Ecran plein
        # self.master.geometry("1200x800") l'un ou l'autre
        self.master.title("Reconnaissance Faciale - Police Scientifique")
        self.master.configure(bg='#f0f0f0')

        # Configuration initiale
        self.title_font = ('Arial', 14, 'bold')
        self.button_font = ('Arial', 10)
        self.primary_color = '#2c3e50'
        self.secondary_color = '#3498db'

        # Chemin des images
        # self.image_folder = "Data bases/Celeb A/Images/img_align_celeba/" # phi : "/Users/phifr/Documents/4A-S1/S2/DvptWeb/img_from_celeba"
        # chemin d'accès dans le git : 
        self.image_folder = "Data Base/selected_images/selected_images"
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
                and int(f.split('.')[0]) >= 200001] # phi <= 100

    def create_widgets(self):
        """Crée les éléments de l'interface"""
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Titre
        self.title_label = ttk.Label(main_frame,
                                text="Veuillez sélectionner une image pour la reconstruction",
                                font=self.title_font,
                                foreground=self.primary_color)
        self.title_label.pack(side = "top", pady=30)

        # Frame des images
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack()

        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side="bottom",pady=20)

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
                    # L'activation sigmoide permet de normaliser les sorties du décodeur pour qu'elles puissent
                    # être interprétées comme des intensités de pixels (normalisées entre 0 et 1),
                    # facilitant ainsi la conversion en image avec PIL
                )

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = ConvAutoencoder().to(self.device)
        # model.load_state_dict(torch.load('conv_autoencoder.pth', map_location=self.device))
        # git
        model.load_state_dict(torch.load('Autoencodeur/conv_autoencoder.pth', map_location=self.device))
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
        # Si "Veuillez sélectionner un portrait" n'est pas visible, le réafficher (après avoir appuyer sur Retour au menu)
        if not self.title_label.winfo_viewable():
            self.title_label.pack(side="top", pady=30, before=self.image_frame) # before=self.image_frame pour forcer le positionnement au top

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

        original_img_frame = ttk.Frame(result_frame)
        original_img_frame.pack(side="left", padx=20)
        ttk.Label(original_img_frame, text="Image originale", font=self.title_font).pack(side="top", pady=30)
        self.display_image(original, original_img_frame, "left")

        reconstructed_img_frame = ttk.Frame(result_frame)
        reconstructed_img_frame.pack(side="right", padx=20)
        ttk.Label(reconstructed_img_frame, text="Image reconstruite", font=self.title_font).pack(side="top", pady=30)
        self.display_image(reconstructed, reconstructed_img_frame, "right")

        # Bouton de retour
        ttk.Button(self.image_frame,
                   text="Retour au menu",
                   command=self.load_new_images).pack(pady=20)

    def process_image(self, img_path):
        """Traite une image via l'autoencodeur"""
        img = Image.open(img_path).convert("RGB")

        # La méthode unsqueeze(0) ajoute une dimension supplémentaire au tenseur, transformant ainsi une image de dimensions
        # (C,H,W) en un tenseur de dimensions. (1,C,H,W). C'est essentiel pour le modèle qui attend une entrée
        # sous forme de batch, même s'il n'y a qu'une seule image
        tensor_img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Étape 1: Encodage vers l'espace latent
            latent_vector = self.model.encode(tensor_img)
            latent_vector = latent_vector.numpy()    # Convertit en np array

            # [SECTION POUR ALGORITHME GÉNÉTIQUE]
            # Ici on pourrait modifier le latent_vector avant décodage
            # Ex: latent_vector = genetic_algorithm(latent_vector)
            
            solutions = algo_genetique_parallel.real_separation(latent_vector[0])

            # Convertir en tenseur PyTorch
            sol = torch.tensor(solutions[0], dtype=torch.float32) # Ne prendre que l'image en position 0, il faudra faire une boucle et afficher les 10 images après

            # Changer la forme en [1, 128, 8, 8]
            sol = sol.view(1, 128, 8, 8)

            # Étape 2: Décodage à partir de l'espace latent
            reconstructed = self.model.decode(sol).cpu().numpy()[0].transpose(1, 2, 0)

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
        # Suppression aussi de la phrase veuillez choisir un portrait à ajouter
        self.title_label.pack_forget()

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
