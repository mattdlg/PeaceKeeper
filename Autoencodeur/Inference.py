import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import random
import os
import torch
import numpy as np
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AlgoGenetique import algo_genetique_parallel as GA
from AlgoGenetique import algo_genetique_multiple_target as GAm
from AlgoGenetique import user_driven_algo_gen as udGA


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
        # self.load_new_images()
        self.load_new_images_with_multiple_selection()

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
                   text="Valider", 
                   command=self.process_multiple_selection).pack(side=tk.LEFT, padx=10)
        
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

    def load_new_images_with_multiple_selection(self):
        # Si "Veuillez sélectionner un portrait" n'est pas visible, le réafficher (après avoir appuyer sur Retour au menu)
        if not self.title_label.winfo_viewable():
            self.title_label.pack(side="top", pady=30, before=self.image_frame) # before=self.image_frame pour forcer le positionnement au top

        for widget in self.image_frame.winfo_children():
            widget.destroy()

        available_images = [img for img in self.all_images if img not in self.used_images]
        selected_images = random.sample(available_images, min(6, len(available_images)))
        self.used_images.update(selected_images)

        self.image_vars = []

        for i, img_name in enumerate(selected_images):
            img_path = os.path.join(self.image_folder, img_name)
            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)

                var = tk.BooleanVar()
                chk = ttk.Checkbutton(self.image_frame, image=photo, variable=var)
                chk.image = photo
                chk.grid(row=i // 3, column=i % 3, padx=10, pady=10)

                self.image_vars.append((var, img_path))
            except Exception as e:
                print(f"Erreur chargement image {img_name}: {e}")   

    def process_selection(self, img_path):
        """Gère la sélection d'image"""
        if messagebox.askyesno("Confirmation", "L'algorithme sera appliqué sur cette image.\nVoulez-vous continuer ?"):
            # self.show_reconstruction(img_path)
            self.show_whole_reconstruction(img_path)

    def process_multiple_selection(self):
        if messagebox.askyesno("Confirmation", "L'algorithme sera appliqué sur cette image.\nVoulez-vous continuer ?"):
            self.selected_images = [img_path for var, img_path in self.image_vars if var.get()]
            # print(len(self.selected_images))
            self.show_whole_reconstruction(self.selected_images)

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
        
    def show_whole_reconstruction(self, img_path):
        # original, reconstructed = self.process_image(img_path)
        originals, reconstructed = self.process_multiple_images(img_path)

        self.clear_interface()

        result_frame = ttk.Frame(self.image_frame)
        result_frame.pack(expand=True, pady=20)

        original_img1_frame = ttk.Frame(result_frame)
        original_img1_frame.grid(row = 0, column = 0, padx=10, pady=5)
        ttk.Label(original_img1_frame, text="Image originale", font=self.title_font).pack(side="top", pady=10)
        self.display_image(originals[0], original_img1_frame, "left")

        original_img2_frame = ttk.Frame(result_frame)
        original_img2_frame.grid(row = 0, column = 2, padx=10, pady=5)
        ttk.Label(original_img2_frame, text="Image originale", font=self.title_font).pack(side="top", pady=10)
        self.display_image(originals[1], original_img2_frame, "left")

        for k in range(len(reconstructed)):
            reconstructed_img_frame = ttk.Frame(result_frame)
            reconstructed_img_frame.grid(row = 1 + k // 3, column = k % 3, padx=5, pady=5)
            # ttk.Label(reconstructed_img_frame, text= f"Image reconstruite {k}", font=self.title_font).pack(side="top", pady=20)
            self.display_image(reconstructed[k], reconstructed_img_frame, "right")

        # Bouton de confirmation de sélection d'images pour relancer algo gen
        ttk.Button(self.image_frame, 
                   text="Valider", 
                   command=self.process_multiple_selection).pack(side=tk.LEFT, padx=10)

        # Bouton de retour
        ttk.Button(self.image_frame,
                   text="Retour au menu",
                   command=self.load_new_images).pack(side=tk.RIGHT, pady=10)
        

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
            
            # solutions = GA.real_separation(latent_vector[0])
            # solutions = GA.real_global(latent_vector[0])
            # solutions = GA.varying_target(latent_vector[0], 6)

            targets_list = GAm.varying_target(latent_vector[0], 6)
            solutions = GAm.ga_multiple_targets_separated(targets_list)

            # print(solutions.shape)

            # Convertir en tenseur PyTorch
            # sol = torch.tensor(solutions[0], dtype=torch.float32) # Ne prendre que l'image en position 0, il faudra faire une boucle et afficher les 10 images après
            sol = torch.tensor(solutions, dtype=torch.float32)

            # Changer la forme en [1, 128, 8, 8]
            sol = sol.view(solutions.shape[0], 128, 8, 8) # 6 images reconstructes

            # Étape 2: Décodage à partir de l'espace latent:

            # .cpu() assure que le tenseur est transféré sur le CPU
            # .numpy() convertit le tenseur PyTorch en un tableau numpy, ce qui facilite la manipulation ultérieure
            # en dehors de PyTorch
            # [0] récupère la première (et ici unique) image du batch
            # .transpose(1, 2, 0) réarrange les dimensions du tableau. Par défaut, PyTorch utilise l'ordre
            # (channels, height, width) tandis que PIL et la plupart des bibliothèques d'affichage d'image attendent
            # l'ordre (height, width, channels)
            
            # reconstructed = self.model.decode(sol).cpu().numpy()[0].transpose(1, 2, 0)
            # print(self.model.decode(sol).cpu().numpy().shape)
            reconstructed = self.model.decode(sol).cpu().numpy().transpose(0, 2, 3, 1)

        img_reconstructed = []
        for k in range(reconstructed.shape[0]):
            img_reconstructed.append(Image.fromarray((reconstructed[k] * 255).astype(np.uint8)))

        # return img, Image.fromarray((reconstructed * 255).astype(np.uint8))
        return img, img_reconstructed

    def process_multiple_images(self, img_paths):
        """Traite une image via l'autoencodeur"""
        list_vectors = []
        list_originals = []
        for path in img_paths :
            img = Image.open(path).convert("RGB")
            list_originals.append(img)

        # La méthode unsqueeze(0) ajoute une dimension supplémentaire au tenseur, transformant ainsi une image de dimensions
        # (C,H,W) en un tenseur de dimensions. (1,C,H,W). C'est essentiel pour le modèle qui attend une entrée
        # sous forme de batch, même s'il n'y a qu'une seule image
            tensor_img = self.transform(img).unsqueeze(0).to(self.device)
            list_vectors.append(tensor_img)

        with torch.no_grad():
            # Étape 1: Encodage vers l'espace latent
            for i in range(len(list_vectors)) :
                list_vectors[i] = self.model.encode(list_vectors[i])
                list_vectors[i] = list_vectors[i][0].numpy()    # Convertit en np array

            print(list_vectors[0].shape)
            # [SECTION POUR ALGORITHME GÉNÉTIQUE]
            # Ici on pourrait modifier le latent_vector avant décodage
            # Ex: latent_vector = genetic_algorithm(latent_vector)

            """targets_list = GAm.create_multiple_target_from_pictures([v[0] for v in list_vectors], 6)
            norm_targets, min_val, max_val = GAm.normalization(np.array(targets_list))
            
            solutions = GAm.run_multiple_ga(norm_targets)
            solutions = GAm.denormalization(solutions, min_val, max_val)"""
            space_limit = np.max(np.asarray(list_vectors))
            solutions = udGA.run_ga(list_vectors, nb_solutions=6, crossover_method="square", 
                                    mutation_rate=0.3, sigma_mutation=0.5)

            # Convertir en tenseur PyTorch
            # sol = torch.tensor(solutions[0], dtype=torch.float32) # Ne prendre que l'image en position 0, il faudra faire une boucle et afficher les 10 images après
            sol = torch.tensor(solutions, dtype=torch.float32)

            # Changer la forme en [1, 128, 8, 8]
            sol = sol.view(solutions.shape[0], 128, 8, 8) # 6 images reconstructes

            # Étape 2: Décodage à partir de l'espace latent:

            # .cpu() assure que le tenseur est transféré sur le CPU
            # .numpy() convertit le tenseur PyTorch en un tableau numpy, ce qui facilite la manipulation ultérieure
            # en dehors de PyTorch
            # [0] récupère la première (et ici unique) image du batch
            # .transpose(1, 2, 0) réarrange les dimensions du tableau. Par défaut, PyTorch utilise l'ordre
            # (channels, height, width) tandis que PIL et la plupart des bibliothèques d'affichage d'image attendent
            # l'ordre (height, width, channels)
            
            # reconstructed = self.model.decode(sol).cpu().numpy()[0].transpose(1, 2, 0)
            # print(self.model.decode(sol).cpu().numpy().shape)
            reconstructed = self.model.decode(sol).cpu().numpy().transpose(0, 2, 3, 1)

        img_reconstructed = []
        for k in range(reconstructed.shape[0]):
            img_reconstructed.append(Image.fromarray((reconstructed[k] * 255).astype(np.uint8)))

        # return img, Image.fromarray((reconstructed * 255).astype(np.uint8))
        return list_originals, img_reconstructed

    # ---------- Utilitaires ----------
    def display_image(self, img, frame, side):
        """Affiche une image dans l'interface"""
        disp_img = ImageTk.PhotoImage(img.resize((200, 200)))
        var = tk.BooleanVar()
        chk = ttk.Checkbutton(frame, image = disp_img, variable = var)
        chk.image = disp_img
        chk.pack(side=side, padx=20)
        # pour relancer ensuite il faudra que ce soit des boutons comme au début

    def clear_interface(self):
        """Nettoie l'interface"""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        # Suppression aussi de la phrase veuillez choisir un portrait à ajouter
        self.title_label.pack_forget()

    def confirm_new_images(self):
        """Confirmation de rechargement"""
        if messagebox.askyesno("Nouvelles Images", "Cela va charger 6 nouvelles images non visionnées.\nContinuer ?"):
            # self.load_new_images()
            self.load_new_images_with_multiple_selection()

    def confirm_exit(self):
        """Confirmation de sortie"""
        if messagebox.askyesno("Quitter", "Êtes-vous sûr de vouloir quitter l'application ?"):
            self.master.destroy()


# ---------- Lancement de l'application ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
