import glob
import os
import sys
import zipfile
import torch
import numpy as np
from PIL import Image
import random
from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
from PyQt6.QtGui import QCursor, QPixmap
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime
from shutil import copyfile
from AlgoGenetique import user_driven_algo_gen as udGA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


try:
    from Autoencodeur.utils_autoencoder import load_best_hyperparameters, Autoencoder, device, transform_
except ImportError:
    from utils_autoencoder import load_best_hyperparameters, Autoencoder, device, transform_


##############################################################################
# 0) Chargement des poids du modèle
##############################################################################

def load_hyperparameters(filename: str = "best_hyperparameters.pth") -> Dict[str, Any]:
    """
    Charge les hyperparamètres du modèle avec gestion automatique des chemins

    Essaie plusieurs emplacements possibles dans cet ordre :
    1. Chemin relatif direct (compatibilité historique)
    2. Dans le dossier du script appelant
    3. Dans le sous-dossier Autoencodeur

    Parameters
    ----------
    filename : str, optional
        Nom du fichier contenant les hyperparamètres (par défaut: "best_hyperparameters.pth")

    Returns
    -------
    Dict[str, Any]
        Dictionnaire des hyperparamètres chargés s'ils sont trouvés, erreur sinon

    """
    # Liste des chemins à tester par ordre de priorité
    search_paths = [
        filename,  # 1. Ancien chemin relatif
        Path(__file__).parent / filename,  # 2. Dossier courant
        Path(__file__).parent.parent / "Autoencodeur" / filename,  # 3. Dossier alternatif
    ]

    last_error = None
    for path in search_paths:
        try:
            path_str = str(path)
            if os.path.exists(path_str):
                params = load_best_hyperparameters(path_str)
                print(f"Chargement réussi depuis : {path_str}")
                return params
        except Exception as e:
            last_error = e
            continue
    print(
        f"- Pour le load des hyperparam: Aucun fichier trouvé aux emplacements :\n"
        f"    1. {search_paths[0]}\n"
        f"    2. {search_paths[1]}\n"
        f"    3. {search_paths[2]}\n"
        f"- Dernière erreur : {str(last_error)}\n"
    )


# Chargement des meilleurs hyperparam
best_params = load_hyperparameters()


##############################################################################
# 1) Fenêtre de Tutoriel (QDialog)
##############################################################################
class TutorielDialog(QtWidgets.QDialog):
    """Fenêtre de tutoriel avec affichage progressif du texte

    Points clés :
    L'intervalle du timer (40ms)
    Les touches clavier spécifiques (Espace/A/Entrée)
    Le système d'indexation du texte
    """

    def __init__(self, sentences, parent=None):
        """Initialise la fenêtre de tutoriel.

        Configure :
        - Une fenêtre modale sans bordure
        - Un fond noir avec bordure blanche
        - Le système d'affichage progressif
        - Le bouton 'Skip tutoriel'
        """
        super().__init__(parent)
        # Fenêtre modale, sans décorations
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setModal(True)
        self.setFixedSize(800, 400)
        self.setStyleSheet("background-color: #000000; border: 5px solid #fff;")
        self.setAutoFillBackground(True)

        self.sentences = sentences
        self.current_sentence = 0
        self.current_text = ""
        self.char_index = 0

        # Layout principal
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Label qui affiche la phrase en cours
        self.text_label = QtWidgets.QLabel("")
        self.text_label.setStyleSheet("color: white; font-size: 20px;")
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label)

        # Bouton skip
        self.skip_btn = QtWidgets.QPushButton("Skip tutoriel")
        self.skip_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #fff;
                padding: 5px;
                font-size: 16px;
                color: white;
                background-color: #444;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        self.skip_btn.clicked.connect(self.accept)
        layout.addWidget(self.skip_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # Timer pour l'effet d'écriture lettre par lettre
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_text)

        # Démarre la 1ère phrase
        self.start_sentence()

        # Empêcher le focus automatique sur le bouton
        self.skip_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def start_sentence(self):
        """Démarre l'affichage d'une nouvelle phrase.

        Actions :
        1. Charge la phrase courante
        2. Réinitialise l'index de caractère
        3. Lance le timer pour l'effet d'écriture
        4. Ferme le dialogue si aucune phrase disponible

        Gère le cas où :
        - Il n'y a plus de phrases à afficher
        """
        if self.current_sentence < len(self.sentences):
            self.current_text = self.sentences[self.current_sentence]
            self.char_index = 0
            self.text_label.setText("")
            # Lance l'écriture
            self.timer.start(40)  # (à ajuster au choix)
        else:
            # Plus de phrases, on ferme
            self.accept()

    def update_text(self):
        """Met à jour l'affichage du texte caractère par caractère.

        Fonctionnement :
        - Ajoute un caractère à chaque appel du timer
        - S'arrête quand toute la phrase est affichée
        - Intervalle du timer : 40ms (configurable)
        """
        if self.char_index < len(self.current_text):
            # Ajoute la lettre suivante
            self.text_label.setText(self.text_label.text() + self.current_text[self.char_index])
            self.char_index += 1
        else:
            # Phrase terminée
            self.timer.stop()

    def keyPressEvent(self, event):
        """Gère les interactions clavier.

        Comportements :
        - Espace/A :
          * Passe à la phrase suivante si l'affichage est terminé
          * Affiche instantanément la phrase en cours sinon
        - Entrée : Permet la navigation entre éléments focusables
        - Autres touches : Ignorées

        Cas particuliers :
        - Gère la fin du tutoriel après la dernière phrase
        - Empêche les interactions non désirées
        """
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_A):
            # Comportement identique pour Espace et A
            if self.char_index >= len(self.current_text):
                if self.current_sentence + 1 < len(self.sentences):
                    self.current_sentence += 1
                    self.start_sentence()
                else:
                    # Dernière phrase terminée : fermer avec accept()
                    self.accept()
            else:
                # Afficher toute la phrase immédiatement
                self.timer.stop()
                self.text_label.setText(self.current_text)
                self.char_index = len(self.current_text)
            event.accept()
        elif key == QtCore.Qt.Key.Key_Enter or key == QtCore.Qt.Key.Key_Return:
            # Permettre la navigation avec Entrée
            self.focusNextChild()
            event.accept()
        elif key == QtCore.Qt.Key_Escape:
            self.accept()
        else:
            event.ignore()


##############################################################################
# 2) Fenêtre de Génération (QDialog) - 10 images
##############################################################################
class GenerationDialog(QtWidgets.QDialog):
    """Class GenerationDialog

    Cette classe permet d’afficher des images, de laisser l’utilisateur en sélectionner,
    et de lancer la reconstruction avec l’algorithme génétique lorsque l’utilisateur
    clique sur “Génération”.
    """

    def __init__(self, image_folder, parent=None):
        """
    Création d'une instance de la classe GenerationDialog.

    Paramètres
    ----------
    image_folder : str
        Chemin vers le dossier contenant les images.
    parent : QWidget, optionnel
        Widget parent (par défaut None).

    Attributs
    ---------
    image_folder : str
        Chemin vers le dossier contenant les images.
    all_image_paths : list
        Liste contenant les chemins d'accès des fichiers présents dans le dossier d'images.
    autoencoder : AutoencoderModel
        Instance utilisée pour l'encodage et le décodage des images.
    selected_images : list
        Contient les chemins des images sélectionnées par l'utilisateur.
    selected_buttons : list
        Contient les références des boutons QPushButton cliqués.
    visualized_images : set
        Ensemble des images déjà affichées à l'utilisateur.
    button_image_map : dict
        Associe chaque bouton à son image correspondante.
    main_layout : QVBoxLayout
        Layout vertical principal de la fenêtre de dialogue.
    initial_section_layout : QVBoxLayout
        Layout contenant les 10 boutons d’image cliquables.
    original_layout : QVBoxLayout
        Layout utilisé pour afficher les deux boutons des images sélectionnées.
    reconstructed_section_layout : QVBoxLayout
        Layout utilisé pour afficher 6 nouveaux portraits avec des variations.
    random_img_section_layout : QVBoxLayout
        Layout utilisé pour afficher 4 images aléatoires.
    final_image_layout : QHBoxLayout
        Layout utilisé pour afficher le portrait final.
    button_layout : QHBoxLayout
        Layout pour les boutons d’action tels que “Valider”, “Nouvelles Images”, “Portrait Final” et “Fermer”.
    self.transforms : torchvision.transforms.Compose
        Composition de transformations de prétraitement appliquées aux images d'entrée avant de les passer dans le modèle.
        Inclut le redimensionnement, le recadrage centré, et la normalisation vers le format tenseur.
    self.device : torch.device
        Spécifie l’appareil (CPU ou GPU) sur lequel les tenseurs et le modèle autoencodeur seront alloués.
        Sélectionne automatiquement 'cuda' si un GPU est disponible ; sinon utilise 'cpu'.
    self.model : Autoencoder
        Le modèle d’autoencodeur convolutif utilisé pour encoder et décoder les images.
        Il est initialisé avec les meilleurs hyperparamètres.
    """

        super().__init__(parent)
        self.setWindowTitle("Variation des portraits")

        self.image_folder = image_folder
        self.all_image_paths = glob.glob(os.path.join(self.image_folder, "*"))
        self.selected_images = []  # Liste pour garder trace des images sélectionnées
        self.selected_buttons = []
        self.visualized_images = set()  # Ensemble pour garder trace des images déjà affichées
        self.button_image_map = {}  # Dictionnaire pour lier les boutons aux images

        # Configuration de la fenêtre
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setModal(True)
        self.setStyleSheet("background-color: #000000; border: none;")
        self.setAutoFillBackground(True)
        self.showFullScreen()
        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

        # Layout principal
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Créer un layout pour afficher 10 images
        self.initial_section_layout = QtWidgets.QVBoxLayout()
        black_block = QtWidgets.QWidget()
        black_block.setStyleSheet("background-color: black;")
        black_block.setFixedHeight(80)
        self.initial_section_layout.addWidget(black_block)
        self.title_label_choice = QtWidgets.QLabel("Veuillez sélectionner 2 portraits")
        self.title_label_choice.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label_choice.setStyleSheet(
            "color: white; font-weight: bold; font-size: 28px; padding: 2px; border: none;")
        self.title_label_choice.setFixedHeight(40)
        self.initial_section_layout.addWidget(self.title_label_choice)
        self.initial_layout = QtWidgets.QGridLayout()
        self.initial_layout.setContentsMargins(0, 70, 0, 50)
        self.initial_section_layout.addLayout(self.initial_layout)
        self.main_layout.addLayout(self.initial_section_layout)

        # Layout pour afficher les images sélectionnées
        self.original_layout = QtWidgets.QVBoxLayout()
        self.title_label = QtWidgets.QLabel("Images sélectionnées")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.title_label.setFixedHeight(40)
        self.original_layout.addWidget(self.title_label)
        self.images_layout = QtWidgets.QGridLayout()
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.original_layout.addLayout(self.images_layout)

        # Layout pour afficher les images reconstruites par l'algorithme génétique
        self.reconstructed_section_layout = QtWidgets.QVBoxLayout()
        self.reconstructed_section_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label_choice = QtWidgets.QLabel("Veuillez sélectionner 2 portraits")
        self.title_label_choice.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label_choice.setStyleSheet(
            "color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.reconstructed_section_layout.addWidget(self.title_label_choice)
        self.reconstructed_layout = QtWidgets.QGridLayout()
        self.reconstructed_layout.setContentsMargins(0, 20, 0, 20)
        self.reconstructed_section_layout.addLayout(self.reconstructed_layout)

        # Créer un layout vertical pour les 2 premiers layout
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.addLayout(self.original_layout)
        self.left_layout.addLayout(self.reconstructed_section_layout)

        # Layout pour afficher des nouvelles images encore jamais vu par l'utilisateur
        self.random_img_section_layout = QtWidgets.QVBoxLayout()
        self.random_img_section_layout.setContentsMargins(0, 0, 0, 0)
        self.title_random_img_section = QtWidgets.QLabel("Nouvelles images")
        self.title_random_img_section.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_random_img_section.setStyleSheet(
            "color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.random_img_section_layout.addWidget(self.title_random_img_section)
        self.random_img_grid_layout = QtWidgets.QGridLayout()
        self.random_img_grid_layout.setContentsMargins(0, 20, 0, 20)
        self.random_img_section_layout.addLayout(self.random_img_grid_layout)

        # Créer un layout horizontal pour ajouter left_layout et random_img_section_layout
        self.all_images_as_btn_layout = QtWidgets.QHBoxLayout()
        self.all_images_as_btn_layout.addLayout(self.left_layout)
        self.all_images_as_btn_layout.addLayout(self.random_img_section_layout)

        # Layout pour afficher le portrait robot définitif
        self.final_reconstruction_layout = QtWidgets.QVBoxLayout()
        black_block2 = QtWidgets.QWidget()
        black_block2.setStyleSheet("background-color: black;")
        black_block2.setFixedHeight(60)
        self.final_reconstruction_layout.addWidget(black_block2)
        self.title_label = QtWidgets.QLabel("Portrait définitif")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-weight: bold; font-size: 28px; padding: 2px; border: none;")
        self.final_reconstruction_layout.addWidget(self.title_label)
        self.final_reconstruction_layout.addSpacing(50)
        self.final_image_layout = QtWidgets.QHBoxLayout()
        self.final_reconstruction_layout.addLayout(self.final_image_layout)

        # Layout pour les boutons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.button_layout.setSpacing(10)
        self.button_layout.setContentsMargins(0, 0, 0, 0)

        # Ajout du bouton 'Fermer'
        self.close_btn = QtWidgets.QPushButton("Fermer")
        self.close_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #fff;
                padding: 5px;
                font-size: 16px;
                color: white;
                background-color: #444;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        self.close_btn.clicked.connect(self.accept)
        self.button_layout.addWidget(self.close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Ajout du bouton 'Valider'
        self.validate_btn = QtWidgets.QPushButton("Valider")
        self.validate_btn.setStyleSheet("""
                                QPushButton {
                                    border: 2px solid #fff;
                                    padding: 5px;
                                    font-size: 16px;
                                    color: white;
                                    background-color: #444;
                                }
                                QPushButton:hover {
                                    background-color: #666;
                                }
                            """)
        self.validate_btn.clicked.connect(self.validate_selection)
        self.button_layout.addWidget(self.validate_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Ajout du bouton "Nouvelles images"
        self.load_new_img = QtWidgets.QPushButton("Nouvelles images")
        self.load_new_img.setStyleSheet(""" QPushButton {
                                border: 2px solid #fff;
                                padding: 5px;
                                font-size: 16px;
                                color: white;
                                background-color: #444;
                            }
                            QPushButton:hover {
                                background-color: #666;
                            }
                        """)
        self.load_new_img.clicked.connect(self.load_images)
        self.button_layout.addWidget(self.load_new_img, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Ajout du bouton "Portrait définitif"
        self.final_btn = QtWidgets.QPushButton("Portrait définitif")
        self.final_btn.setStyleSheet("""
                    QPushButton {
                        border: 2px solid #fff;
                        padding: 5px;
                        font-size: 16px;
                        color: white;
                        background-color: #444;
                    }
                    QPushButton:hover {
                        background-color: #666;
                    }
                """)
        self.final_btn.clicked.connect(self.display_definitive_portrait)

        self.main_layout.addLayout(self.button_layout)

        # Initialisation du modèle
        self.device = device
        self.model = Autoencoder(nb_channels=best_params['nb_channels'], nb_layers=best_params['nb_layers']).to(
            self.device)

        try:
            self.load_model()
        except FileNotFoundError as e:
            print(f"ERREUR CRITIQUE: {e}")

        self.transforms = transform_

        # Charger les images
        self.load_images()

    def load_model(self):
        """
        Tente de charger le modèle depuis plusieurs emplacements possibles
        en suivant un ordre de priorité défini. Si aucun fichier valide n'est trouvé,
        lève une exception FileNotFoundError avec la liste des chemins testés.

        Ordre de recherche :
        1. Chemin relatif direct (pour compatibilité avec les anciennes versions)
        2. Dossier contenant le script appelant
        3. Sous-dossier Autoencodeur dans le parent du script

        Returns
        -------
        None


        Raises
        ------
        FileNotFoundError
            Si aucun fichier valide n'est trouvé dans les emplacements testés
        RuntimeError
            Si le chargement échoue pour des raisons de compatibilité des poids

        Notes
        -----
        - Les messages de debug sont affichés dans la console lors des tentatives
        - Le map_location est automatiquement géré en fonction de self.device
        - Format de fichier attendu : .pth (format PyTorch)
        """
        # Liste des chemins possibles (ordre de priorité)
        possible_paths = [
            'conv_autoencoder.pth',  # 1. Ancien chemin (pour compatibilité)
            Path(__file__).parent / 'conv_autoencoder.pth',  # 2. Même dossier
            Path(__file__).parent.parent / 'Autoencodeur' / 'conv_autoencoder.pth',  # 3. Chemin alternatif
        ]

        for path in possible_paths:
            path_str = str(path)
            if os.path.exists(path_str):
                try:
                    self.model.load_state_dict(torch.load(path_str, map_location=self.device))
                    print(f"Poids chargés depuis: {path_str}")
                    return
                except Exception as e:
                    print(f"Erreur lors du chargement depuis {path_str}: {e}")
                    continue

        # Si aucun chemin n'a fonctionné
        raise FileNotFoundError(
            f"Aucun fichier de poids valide trouvé. Chemins testés:\n"
            f"1. {possible_paths[0]}\n"
            f"2. {possible_paths[1]}\n"
            f"3. {possible_paths[2]}"
        )

    def reset_selected_buttons(self):
        """
        Réinitialise la liste des boutons et images sélctionnés et retire l'effet
        d'ombrage de ces images pour prévenir qu'elles sont déselctionnées.
        Cette fonction est appelée si des images ont été sélectionnées et que l'utilisateur
        décide ensuite de  générer 10 nouvelles images.

        Makes use of
        ------------
        selected_images : list
            Contient le chemin des images sélectionnées par l'utilisateur.
        selected_buttons : list
            Contient les références aux QPushButton cliqués (associés aux images).

        Returns
        -------
        None
        """
        for button in self.selected_buttons:
            button.setGraphicsEffect(None)

        # Réinitialiser les listes
        self.selected_buttons.clear()
        self.selected_images.clear()
        print("Sélections réinitialisées")

    def load_images(self):
        """
        Sélectionne jusqu’à 10 images qui n'ont pas encore été visualisées depuis le dossier spécifié.
        Chaque image est affichée sous forme de bouton grâce à la méthode 'display_buttons'.
        Une fois une image chargée, elle est marquée comme visualisée.
        Si aucune image non visualisée ne reste, une boîte de message d'avertissement est affichée à l’utilisateur.

        Les images sont redimensionnées pour assurer une cohérence visuelle et les boutons sont ajoutés à une grille 5x2.
        Chaque bouton permet à l’utilisateur de sélectionner des images pour des interactions ultérieures.

        Paramètres
        ----------
        None

        Utilise
        -------
        self.selected_buttons : list
            Contient les références des instances de QPushButton cliquées.
        self.image_folder : str
            Chemin vers le dossier contenant les images.
        self.visualized_images : set
            Ensemble des images déjà affichées à l'utilisateur.
        self.initial_layout : QtWidgets.QGridLayout
            Layout dans lequel les boutons des images sont ajoutés.
        self.display_buttons : méthode
            Méthode utilisée pour ajouter les images sous forme de boutons dans un layout donné.
        self.reset_selected_buttons : méthode
            Réinitialise la liste des images sélectionnées et retire l’effet d’ombre sur ces images.

        Retourne
        --------
        None
        """

        if self.selected_buttons:
            self.reset_selected_buttons()

        remaining_images = [img_path for img_path in self.all_image_paths if img_path not in self.visualized_images]

        if not remaining_images:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Avertissement")
            msg_box.setText("Impossible de générer de nouvelles images, toutes les images ont été visualisées")

            msg_box.setMinimumSize(500, 300)
            msg_box.setStyleSheet("""
                                            QMessageBox {
                                                background-color: black;  /* Fond noir */
                                                border: none;  /* Supprime les bordures */
                                            }
                                            QLabel {
                                                color: white;  /* Texte en blanc */
                                                font-size: 14px;  /* Taille de police */
                                                font-weight: bold;  /* Mettre en gras (optionnel) */
                                                border: none;
                                            }
                                            QPushButton {
                                                background-color: gray;  /* Boutons en gris */
                                                color: white;  /* Texte des boutons en blanc */
                                                padding: 5px;  /* Réduit les marges internes */
                                                border-radius: 10px;
                                                border: none;
                                            }
                                            QPushButton:hover {
                                                background-color: lightgray; /* Effet survol */
                                            }
                                        """)
            msg_box.exec()
            return

        # Sélectionne 10 images randoms encore jamais visualisées.
        image_paths = random.sample(remaining_images, min(10, len(remaining_images)))

        for i, img_path in enumerate(image_paths):
            self.visualized_images.add(img_path)

        self.display_buttons(self.initial_layout, image_paths, 5, 150)

    def display_original_images(self):
        """
        Affiche les 2 images sélectionnées dans le layout.

        Efface le contenu actuel du layout utilisé pour afficher les images précédemment sélectionnées
        (seulement les images, pas le label "Images sélectionnées"),
        puis affiche les nouvelles images sélectionnées par l’utilisateur. Le layout est ensuite mis à jour
        avec ces images, organisées en deux colonnes à l’aide de la méthode 'display_buttons'.

        Paramètres
        ----------
        None

        Utilise
        -------
        self.selected_images : list
            Contient les références des instances de QPushButton cliquées.
        self.images_layout : QtWidgets.QGridLayout
            Layout dans lequel les boutons des images sont ajoutés.
        self.display_buttons : méthode
            Méthode utilisée pour ajouter les images sous forme de boutons dans un layout donné.

        Retourne
        --------
        None
        """

        for i in reversed(range(self.images_layout.count())):
            widget = self.images_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.display_buttons(self.images_layout, self.selected_images, 2, 150)

    def display_buttons(self, layout, data_img, nb_col, target_size):
        """
        Affiche une liste d’images sous forme de boutons cliquables dans le layout spécifié.

        Gère à la fois les chemins de fichiers image pour une première visualisation (en tant que chaînes de caractères,
        les images invalides étant remplacées par un espace réservé gris), et les tableaux NumPy (pour les images reconstruites).
        Pour chaque image, un QPushButton est créé avec l’image comme icône. Ces boutons sont ensuite organisés
        dans le layout spécifié selon une grille avec un nombre de colonnes défini. Chaque bouton est relié
        à une fonction de rappel permettant la sélection d’image, et est stocké dans le dictionnaire 'button_image_map'
        afin de conserver l’image associée à chaque bouton.

        Paramètres
        ----------
        layout : QGridLayout
            Layout dans lequel les boutons des images seront ajoutés.
        data_img : list
            Liste d’images à afficher. Chaque élément peut être un chemin de fichier ou un tableau NumPy.
        nb_col : int
            Nombre de colonnes à utiliser pour organiser les boutons dans le layout.
        target_size : int
            Taille cible des images dans le layout.

        Utilise
        -------
        self.button_image_map : dict
            Associe chaque bouton à son image correspondante.
        self.select_image_from_generated : méthode
            Méthode utilisée pour mettre à jour les images sélectionnées par l'utilisateur.

        Retourne
        --------
        None
            Cette méthode modifie directement le layout.
        """

        for i, img_array in enumerate(data_img):
            if isinstance(img_array, str):
                img_qpixmap = QtGui.QPixmap(img_array)
                if img_qpixmap.isNull():
                    img_qpixmap = QtGui.QPixmap(target_size, target_size)
                    img_qpixmap.fill(QtGui.QColor("gray"))
            elif isinstance(img_array, np.ndarray):
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(
                    img_pil.tobytes("raw", "RGB"),
                    img_pil.width, img_pil.height, img_pil.width * 3, QtGui.QImage.Format.Format_RGB888
                ))
            img_qpixmap = img_qpixmap.scaled(target_size, target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
            img_width = img_qpixmap.width()
            img_height = img_qpixmap.height()

            # Création du bouton
            btn = QtWidgets.QPushButton(self)
            btn.setFixedSize(img_width, img_height)
            btn.setIcon(QtGui.QIcon(img_qpixmap))
            btn.setIconSize(QtCore.QSize(img_width, img_height))
            btn.setStyleSheet("border: none;")
            btn.clicked.connect(lambda checked, p=img_array, b=btn: self.select_image_from_generated(b))

            # Association de l'image à son bouton dans le dictionnaire
            self.button_image_map[btn] = img_array

            layout.addWidget(btn, i // nb_col, i % nb_col)

    def display_images(self, layout, img_size):
        """
        Affiche les images sélectionnées dans le layout spécifié.
        Cette méthode parcourt 'self.selected_images', qui peut contenir soit
        des chemins de fichiers (str), soit des tableaux NumPy représentant des images.

        Paramètres
        ----------
        layout : QGridLayout
            Layout dans lequel les images seront affichées.
        img_size : int
            Taille cible (img_size = largeur = hauteur) pour les images affichées.

        Utilise
        -------
        self.selected_images : list
            Contient les chemins des images sélectionnées par l’utilisateur.

        Retourne
        --------
        None
            La méthode modifie le layout donné en y ajoutant les images.
        """

        for img_data in self.selected_images:
            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: none; background: none;")

            if isinstance(img_data, str):
                original_img = QtGui.QPixmap(img_data)
            elif isinstance(img_data, np.ndarray):
                img_pil = Image.fromarray((img_data * 255).astype(np.uint8))
                original_img = QtGui.QPixmap.fromImage(QtGui.QImage(
                    img_pil.tobytes("raw", "RGB"),
                    img_pil.width, img_pil.height, img_pil.width * 3,
                    QtGui.QImage.Format.Format_RGB888
                ))
            label.setPixmap(original_img.scaled(img_size, img_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def display_generated_images(self, generated_images):
        """
        Cette méthode est appelée après l’exécution de l’algorithme génétique pour générer
        de nouvelles variations d’images. Elle commence par supprimer les anciens QPushButtons d’images affichées.
        Ensuite, elle affiche les nouvelles images à l’aide de la méthode 'display_buttons' dans un layout en grille à 3 colonnes.

        Paramètres
        ----------
        generated_images : list
            Liste d’images représentant des variations à partir des images sélectionnées.

        Utilise
        -------
        self.reconstructed_layout : QGridLayout
            Layout où les nouvelles images générées seront affichées.
        self.validate_btn : QPushButton
            Bouton de validation permettant de soumettre le choix de l’utilisateur.
        self.display_buttons : méthode
            Méthode utilisée pour ajouter les images sous forme de boutons dans le layout reconstruit.

        Retourne
        --------
        None
            La méthode modifie le layout reconstruit.
        """

        # Nettoyer l'affichage actuel
        # Suppression des boutons sauf "Valider", "Fermer"
        for i in reversed(range(self.reconstructed_layout.count())):
            widget = self.reconstructed_layout.itemAt(i).widget()
            if widget and isinstance(widget, QtWidgets.QPushButton):
                if widget not in [self.validate_btn]:
                    widget.deleteLater()

        self.display_buttons(self.reconstructed_layout, generated_images, 3, 150)

    def display_new_randoms_images(self):
        """
        Affiche 3 images randoms encore jamais vu par l'utilisateur.

        Nettoie le contenu actuel du layout (uniquement les précédentest images).
        Ces images sont des boutons cliquables.
        Le layout est mis à jour avec 3 images organisées en format colonne,
        en utilisant la méthode 'display_buttons'.

        Paramètres
        ----------
        None

        Utilise
        -------
        self.random_img_grid_layout : list
            Layout dans lequel les boutons d’images sont ajoutés.
        self.display_buttons : méthode
            Méthode utilisée pour ajouter les images sous forme de boutons dans un layout donné.
        self.all_image_paths : list
            Liste contenant les chemins d’accès des fichiers présents dans le dossier Images.
        self.visualized_images : set
            Ensemble des images déjà affichées à l’utilisateur.

        Retourne
        --------
        None
        """

        for i in reversed(range(self.random_img_grid_layout.count())):
            widget = self.random_img_grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        remaining_images = [img_path for img_path in self.all_image_paths if img_path not in self.visualized_images]


        if not remaining_images:
            print("Toutes les img ont été affichées")
            return

        image_paths = random.sample(remaining_images, min(3, len(remaining_images)))

        for i, img_path in enumerate(image_paths):
            self.visualized_images.add(img_path)

        self.display_buttons(self.random_img_grid_layout, image_paths, 1, 150)

    def display_definitive_portrait(self):
        """
        Affiche le portrait définitif basé sur l’image sélectionnée.

        Cette méthode vérifie d’abord si une seule image a été sélectionnée. Si ce n’est pas le cas,
        elle affiche une boîte de message d’avertissement demandant à l’utilisateur de sélectionner une seule image.
        Si une image est bien sélectionnée, elle nettoie le layout principal
        (en supprimant les boutons inutiles) et met à jour le layout de reconstruction final
        pour afficher le portrait définitif.

        Paramètres
        ----------
        None

        Utilise
        -------
        self.selected_images : list
            Contient les références des instances de QPushButton cliquées.
            La liste doit contenir exactement un élément ici.
        self.selected_buttons : list
            Liste des boutons sélectionnés, réinitialisée au début de la méthode.
        self.button_layout : QHBoxLayout
            Layout contenant les boutons d’action.
        self.main_layout : QVBoxLayout
            Layout vertical principal de la fenêtre de dialogue.
        self.final_reconstruction_layout : QVBoxLayout
            Layout utilisé pour afficher le portrait final.
        self.display_images : méthode
            Affiche les images dans le layout avec une taille spécifiée.
        self.save_definitive_portrait : méthode
            Sauvegarde l’image du portrait définitif sélectionné par l’utilisateur dans le dossier spécifié.

        Retourne
        --------
        None
            La méthode modifie directement le layout de reconstruction final et l’interface utilisateur.
        """

        self.selected_buttons.clear()

        if len(self.selected_images) != 1:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Erreur")
            msg_box.setText("Veuillez sélectionner exactement une image")

            msg_box.setMinimumSize(500, 300)
            msg_box.setStyleSheet("""
                                QMessageBox {
                                    background-color: black;  /* Fond noir */
                                    border: none;  /* Supprime les bordures */
                                }
                                QLabel {
                                    color: white;  /* Texte en blanc */
                                    font-size: 14px;  /* Taille de police */
                                    font-weight: bold;  /* Mettre en gras (optionnel) */
                                    border: none;
                                }
                                QPushButton {
                                    background-color: gray;  /* Boutons en gris */
                                    color: white;  /* Texte des boutons en blanc */
                                    padding: 5px;  /* Réduit les marges internes */
                                    border-radius: 10px;
                                    border: none;
                                }
                                QPushButton:hover {
                                    background-color: lightgray; /* Effet survol */
                                }
                            """)
            msg_box.exec()
            return

        self.button_layout.removeWidget(self.validate_btn)
        self.validate_btn.deleteLater()
        self.button_layout.removeWidget(self.final_btn)
        self.final_btn.deleteLater()
        self.main_layout.removeItem(self.button_layout)
        self.remove_layout(self.main_layout, self.original_layout)
        self.remove_layout(self.main_layout, self.reconstructed_section_layout)
        self.remove_layout(self.main_layout, self.random_img_section_layout)
        self.main_layout.addLayout(self.final_reconstruction_layout)
        self.main_layout.addLayout(self.button_layout)
        self.display_images(self.final_image_layout, 400)

        self.save_definitive_portrait()

    def generate_new_images(self, selected_images):
        """
        Génère 6 nouvelles images à partir des deux images sélectionnées en utilisant un algorithme génétique.

        Cette méthode traite les images sélectionnées en les encodant d’abord en vecteurs latents
        à l’aide d’un modèle d’autoencodeur. Ensuite, elle applique un algorithme génétique pour créer
        6 nouvelles images à partir de ces vecteurs encodés. Enfin, les nouvelles images sont décodées (reconstruites)
        et affichées grâce à la méthode 'display_generated_images'.

        Paramètres
        ----------
        selected_images : list
            Liste contenant les chemins des images sélectionnées (sous forme de chaînes de caractères)
            ou des tableaux NumPy représentant les images sélectionnées. Ces images seront utilisées pour générer les nouvelles.

        Utilise
        -------
        self.transforms : torchvision.transforms.Compose
            Une composition de transformations de prétraitement appliquées aux images avant
            de les passer dans le modèle. Comprend redimensionnement, recadrage centré et normalisation
            au format tensor.
        self.device : torch.device
            Spécifie l’appareil sur lequel les tenseurs et le modèle d’autoencodeur seront alloués.
            Sélectionne automatiquement 'cuda' si un GPU est disponible, sinon utilise 'cpu'.
        self.model : Autoencoder
            Le modèle d’autoencodeur convolutionnel utilisé pour encoder et décoder les images.
            Il est initialisé avec les meilleurs hyperparamètres.
        self.selected_images : list
            Contient les références des instances de QPushButton cliquées
            (utilisées comme entrée pour l’algorithme génétique).
        self.button_layout : QHBoxLayout
            Layout contenant les boutons d’action.
        self.final_btn : QPushButton
            Le bouton "Portrait définitif", affiché une fois les nouvelles images générées,
            permettant de choisir l’image finale.
        self.display_original_images : méthode
            Affiche les images sélectionnées par l’utilisateur.
        self.display_generated_images : méthode
            Affiche les nouvelles images générées par l’algorithme génétique.

        Retourne
        --------
        None
            La méthode modifie l’interface utilisateur en affichant à la fois les images sélectionnées
            et les images générées.
        """

        list_vectors = []

        for img_data in selected_images:
            # Si c'est un chemin de fichier (les 10 images générées initialement):
            if isinstance(img_data, str):
                img = Image.open(img_data).convert("RGB")
            # Si c'est une liste de numpy array (les images regénérées),
            # convertir en PIL.Image
            elif isinstance(img_data, np.ndarray):
                img = Image.fromarray((img_data * 255).astype(np.uint8))
            else:
                print(f"Type de donnée inattendu : {type(img_data)}")
                continue

            tensor_img = self.transforms(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                latent_vector = self.model.encode(tensor_img)
                list_vectors.append(latent_vector[0].cpu().numpy())

        # Appliquer l'algorithme génétique pour générer 6 nouvelles images
        solutions = udGA.run_ga(list_vectors, nb_solutions=6, crossover_method="single-point", mutation_rate=0.3,
                                sigma_mutation=0.25)
        # Convertir en tenseur PyTorch
        sol = torch.tensor(solutions, dtype=torch.float32)
        sol = sol.view(solutions.shape[0], list_vectors[0].shape[0])

        with torch.no_grad():
            reconstructed = self.model.decode(sol).cpu().numpy().transpose(0, 2, 3, 1)

        if self.button_layout.indexOf(self.final_btn) == -1:  # Vérifie si le bouton est déjà dans le layout
            self.button_layout.addWidget(self.final_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # Affiche les images sléectionnées, les images générées par GA et les images randoms
        self.display_original_images()
        self.selected_images.clear()
        self.display_generated_images(reconstructed)
        self.display_new_randoms_images()

    def validate_selection(self):
        """
        Valide la sélection de deux images et affiche l’image reconstruite de la première image sélectionnée.

        Cette méthode s’assure que exactement deux images ont été sélectionnées par l’utilisateur.
        Une fois la sélection validée, le layout principal est mis à jour en supprimant
        le layout initial avec 10 images et en affichant la version reconstruite
        avec un layout contenant les images sélectionnées et les images générées par l'algorithme génétique (GA).
        La méthode 'generate_new_images' est appelée.

        Paramètres
        ----------
        None

        Utilise
        -------
        self.selected_images : list
            Contient les références des instances de QPushButton cliquées.
            La liste doit contenir exactement deux éléments ici.
        self.main_layout : QVBoxLayout
            Le layout principal de la fenêtre.
        self.reconstructed_section_layout : QVBoxLayout
            Layout utilisé pour afficher 6 nouveaux portraits avec des variations.
        self.button_layout : QHBoxLayout
            Layout des boutons d’action, qui est mis à jour lors du processus de validation.
        self.load_new_img : QPushButton
            Un bouton pour générer 10 nouveaux portraits.
            Il est supprimé s’il était présent.
        self.original_layout : QVBoxLayout
            Layout utilisé pour afficher les deux boutons des images sélectionnées.
        self.generate_new_images : méthode
            Méthode utilisée pour générer et afficher les images grâce à l’algorithme génétique à partir des images sélectionnées.

        Retourne
        --------
        None
            La méthode modifie l’interface utilisateur en affichant les images sélectionnées reconstruites.
        """

        self.selected_buttons.clear()

        if len(self.selected_images) != 2:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Erreur")
            msg_box.setText("Veuillez sélectionner exactement deux images")

            msg_box.setMinimumSize(500, 300)
            msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: black;  /* Fond noir */
                        border: none;  /* Supprime les bordures */
                    }
                    QLabel {
                        color: white;  /* Texte en blanc */
                        font-size: 14px;  /* Taille de police */
                        font-weight: bold;  /* Mettre en gras (optionnel) */
                        border: none;
                    }
                    QPushButton {
                        background-color: gray;  /* Boutons en gris */
                        color: white;  /* Texte des boutons en blanc */
                        padding: 5px;  /* Réduit les marges internes */
                        border-radius: 10px;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: lightgray; /* Effet survol */
                    }
                """)
            msg_box.exec()
            return

        self.remove_layout(self.main_layout, self.initial_section_layout)
        if self.load_new_img in [self.button_layout.itemAt(i).widget() for i in range(self.button_layout.count())]:
            self.button_layout.removeWidget(self.load_new_img)
            self.load_new_img.deleteLater()
        self.main_layout.removeItem(self.button_layout)
        self.main_layout.addLayout(self.all_images_as_btn_layout)
        self.main_layout.addLayout(self.button_layout)
        self.generate_new_images(self.selected_images)

    def select_image_from_generated(self, button):
        """
        Sélectionne un bouton (représentant une image générée par l'algorithme génétique) parmi ceux affichés
        et le met en surbrillance avec un effet d'ombre.

        Cette méthode associe une image générée par l'algorithme génétique à son bouton correspondant et
        met en surbrillance le bouton lorsqu'il est sélectionné en appliquant un effet d'ombre. Si le
        bouton est déjà sélectionné, la sélection est supprimée et le bouton est dé-sélectionné.

        Paramètres
        ----------
        button : QPushButton
            Le bouton représentant l'image générée.

        Utilise
        -------
        self.selected_buttons : list
            Liste des boutons sélectionnés.

        self.selected_images : list
            Liste qui stocke les chemins d'accès des images sélectionnées. Cette liste est mise à jour
            lorsqu'une image est sélectionnée ou désélectionnée.

        self.button_image_map : dict
            Associe chaque bouton à son image correspondante.

        Retourne
        --------
        None
            La méthode met directement à jour la liste des images sélectionnées.

        Notes
        -----
        L'utilisation de selected_buttons est nécessaire car il est impossible de rechercher une correspondance exacte
        dans une liste de np.ndarray (l'image elle-même).
        """

        associated_img = self.button_image_map.get(button)
        if button not in self.selected_buttons:
            self.selected_buttons.append(button)
            if associated_img is not None and id(associated_img) not in [id(img) for img in self.selected_images]:
                self.selected_images.append(associated_img)
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(40)
            shadow.setColor(QtGui.QColor(160, 160, 160))
            shadow.setOffset(0, 0)
            button.setGraphicsEffect(shadow)
        else:
            self.selected_buttons.remove(button)
            self.selected_images = [img for img in self.selected_images if id(img) != id(associated_img)]
            button.setGraphicsEffect(None)

    def save_definitive_portrait(self):
        """
            Sauvegarde l'image du portrait définitif sélectionné par l'utilisateur dans le dossier spécifié.

            Cette méthode vérifie d'abord si le dossier 'ConfirmedSuspects' existe. Si le dossier n'est pas trouvé,
            l'image ne sera pas sauvegardée.
            Si le dossier existe, l'image est sauvegardée avec un nom de fichier généré à partir de la date et l'heure actuelles.
            La méthode prend en compte les types d'images suivantes : chemin str ou tableau NumPy.

            Un message de succès est affiché après un délai de 2 secondes pour informer l'utilisateur du succès de l'opération.

            Utilise
            -------
            self.selected_images : list
                Une liste qui contient les chemins des images sélectionnées.
                Ici il n'y a qu'une image.
            self.get_output_directory : method
                Méthode qui retourne le dossier dans lequel sauvegarder le portrait définitif.
            self.show_success_message : method
                Affiche le message de succès de la sauvegarde.

            Retourne
            -------
            None
                Cette méthode modifie directement l'interface utilisateur en affichant un message de succès.
            """
        definitive_img = self.selected_images[0]

        output_dir = self.get_output_directory()
        if output_dir is None:
            return

        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"suspect_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)

        # Sauvegarder selon le type d'image
        if isinstance(definitive_img, str):
            # Si l’image vient d’un fichier, on copie directement
            copyfile(definitive_img, save_path)
        elif isinstance(definitive_img, np.ndarray):
            # Sinon on convertit en image et on sauve
            img = Image.fromarray((definitive_img * 255).astype(np.uint8))
            img.save(save_path)
        else:
            print("Format d'image non supporté pour l'enregistrement")
            return

        QtCore.QTimer.singleShot(2000, lambda: self.show_success_message(filename,
                                                                         output_dir))  # délai d'affichage du message de 2s

    def show_success_message(self, filename, output_dir):
        """
        Affiche le message de succès de la sauvegarde.

        Paramètres
        ----------
        filename : str
            Nom du fichier image
        output_dir : str
            Nom du dossier de la sauvegarde

        Retours
        -------
        None
            La boîte de dialogue est affichée directement par la fonction et n'a donc pas besoin d'être retourné.
        """
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Sauvegarde réussie")
        msg_box.setText(f"Le portrait {filename} a été enregistré dans : {output_dir}")

        msg_box.setMinimumSize(500, 300)

        msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: black; 
                    border: none; 
                }
                QLabel {
                    color: white;  
                    font-size: 14px;  
                    font-weight: bold;  
                    border: none;
                }
                QPushButton {
                    background-color: gray;  
                    color: white; 
                    padding: 5px;  
                    border-radius: 10px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: lightgray; 
                }
            """)
        msg_box.exec()

    def get_output_directory(self):
        """
        Détermine dynamiquement le dossier dans lequel sauvegarder le portrait définitif.

        Retourne
        -------
        dir_path :
            Le chemin du dossier ConfirmedSuspects, existant ou nouvellement créé.
        """
        possible_dirs = [
            Path('ConfirmedSuspects'),
            Path(__file__).parent / 'ConfirmedSuspects',
            Path(__file__).parent.parent / 'ConfirmedSuspects',
        ]

        for dir_path in possible_dirs:
            if dir_path.exists():
                return dir_path

        # Si ConfirmedSuspects n'est pas trouvé
        print("ConfirmedSuspects introuvable")
        return None

    def remove_layout(self, layout_parent, layout):
        """
        Supprime un layout ainsi que tous ses widgets du layout parent donné.

        Paramètres
        ----------
        layout_parent : QLayout
            Le layout parent duquel le layout cible sera supprimé.

        layout : QLayout
            Le layout à supprimer.

        Retourne
        --------
        Aucun
            La méthode modifie directement la structure du layout en supprimant le layout spécifié
            ainsi que ses widgets.
        """

        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)

                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    self.remove_layout(layout, item.layout())

            layout_parent.removeItem(layout)


##############################################################################
# 3)Background Vidéo
##############################################################################
class BackgroundVideoWidget(QtWidgets.QGraphicsView):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(
            "border: none; background: transparent;")  # Fond transparent pour ne pas cacher les autres widgets
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                          True)  # Permet aux événements souris de passer à travers

        # Création d'une scène graphique
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)

        # Ajout de l'élément vidéo
        self.video_item = QtMultimediaWidgets.QGraphicsVideoItem()
        self.scene.addItem(self.video_item)

        # Configuration du player
        self.player = QtMultimedia.QMediaPlayer(self)
        self.player.setVideoOutput(self.video_item)
        self.player.setSource(QtCore.QUrl.fromLocalFile(video_path))
        self.player.setLoops(QtMultimedia.QMediaPlayer.Loops.Infinite)  # Boucle infinie

        # Ajout de la sortie audio
        self.audio_output = QtMultimedia.QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.3)  # Volume réglable (de 0.0 à 1.0)

        self.player.play()

    def resizeEvent(self, event):
        """ Ajuste la vidéo à la taille du widget """
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        self.video_item.setSize(QtCore.QSizeF(self.width(), self.height()))
        super().resizeEvent(event)


##############################################################################
# 4) Page Splash (fade in + fade out du texte)
##############################################################################
class SplashPage(QtWidgets.QWidget):
    """Page de splash avec effets de fondu entrant/sortant
    Points clés:
    Système d'animation :
        Deux animations PropertyAnimation pour les fades
        Courbe InOutQuad pour un effet smooth
        Timer pour la transition automatique

    Rendu du texte :
        Positionnement mathématiquement centré
        Utilisation de FontMetrics pour précision
        Effet "texte creux" avec contour+remplissage

    Signalisation :
        Émission du signal à la fin du fade out
        Permet de chaîner des événements

    Valeurs remarquables :
        Durées (1500ms, 3000ms)
        Taille police (72pt)
        Épaisseur contour (5px)
    """

    fade_out_done = QtCore.pyqtSignal()
    """Signal émis quand l'animation de disparition est terminée"""

    def __init__(self, parent=None):
        """
        Initialise l'écran de splash avec animations.

        Args :
            parent (QWidget, optional) : Widget parent. Par défaut None.

        Configuration :
        - Fond noir
        - Effet d'opacité animé
        - Texte centré avec style
        - Séquence automatique fade in -> pause -> fade out
        """
        super().__init__(parent)
        # Configuration visuelle de base
        self.setStyleSheet("background-color: #000000;")
        self.setAutoFillBackground(True)  # Remplissage solide

        # Setup de l'effet d'opacité
        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)  # Commence invisible

        # Animation d'apparition (fade in)
        self.anim_in = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim_in.setDuration(1500)  # 1.5 seconde
        self.anim_in.setStartValue(0)  # Complètement transparent
        self.anim_in.setEndValue(1)  # Complètement opaque
        self.anim_in.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)  # Courbe de progression
        self.anim_in.start()  # Démarre immédiatement

        # Programme le fade out après 3 secondes (3000ms)
        QtCore.QTimer.singleShot(3000, self.start_fade_out)

    def start_fade_out(self):
        """
        Lance l'animation de disparition (fade out).

        Configuration :
        - Durée : 1.5 seconde
        - De opaque (1) à transparent (0)
        - Même courbe de progression que le fade in
        - Émet le signal fade_out_done à la fin
        """
        self.anim_out = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim_out.setDuration(1500)
        self.anim_out.setStartValue(1)
        self.anim_out.setEndValue(0)
        self.anim_out.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
        self.anim_out.finished.connect(lambda: self.fade_out_done.emit())
        self.anim_out.start()

    def paintEvent(self, event):
        """
        Dessine le contenu du splash (texte avec effets).

        Techniques :
        - Texte vectoriel (QPainterPath)
        - Contour noir épais (5px)
        - Remplissage blanc
        - Antialiasing pour lissage
        - Positionnement centré précis
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)  # Lissage activé
        rect = self.rect()  # Dimensions du widget

        # Configuration du texte
        text = "Agency studio"
        font = QtGui.QFont("Arial", 72, QtGui.QFont.Weight.Bold)  # Police gras 72pt

        # Calcul positionnement précis
        fm = QtGui.QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)
        text_height = fm.ascent()  # Hauteur au-dessus de la baseline

        # Centrage horizontal et vertical
        x = (rect.width() - text_width) / 2
        y = (rect.height() + text_height) / 2  # Centre vertical basé sur l'ascent

        # Création du chemin vectoriel
        path = QtGui.QPainterPath()
        path.addText(x, y, font, text)

        # Dessin du contour (ombre/contour noir)
        pen = QtGui.QPen(QtGui.QColor("#000000"), 5)  # Contour noir 5px
        painter.strokePath(path, pen)

        # Remplissage du texte (blanc)
        painter.fillPath(path, QtGui.QColor("white"))


##############################################################################
# 5) Page d'accueil (menu latéral + titre + nouveau background Cyberpunk)
##############################################################################
class HomePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Définition du chemin de la vidéo
        video_dir = Path(__file__).parent.parent / "Elements graphiques" / "Background"
        video_path = str(video_dir / "night-walk-cyberpunk-city-pixel-moewalls-com.mp4")

        # Avec vérification et fallback
        if not os.path.exists(video_path):
            print(f"ERREUR: Fichier vidéo introuvable - {video_path}")
            # Fallback optionnel
            video_path = ""  # ou un chemin par défaut

        self.background = BackgroundVideoWidget(video_path)

        # Stack pour gérer l'empilement
        stacked = QtWidgets.QStackedLayout()
        stacked.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
        stacked.addWidget(self.background)

        # Overlay des éléments interactifs
        overlay = QtWidgets.QWidget()
        overlay_layout = QtWidgets.QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)

        self.menu_widget = QtWidgets.QWidget()
        self.menu_widget.setFixedWidth(220)
        self.menu_widget.setStyleSheet("background-color: rgba(0, 0, 0, 200);")
        menu_layout = QtWidgets.QVBoxLayout(self.menu_widget)
        menu_layout.setContentsMargins(15, 15, 15, 15)
        menu_layout.setSpacing(15)

        menu_label = QtWidgets.QLabel("Menu")
        menu_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        menu_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        menu_layout.addWidget(menu_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        menu_layout.addStretch(1)

        self.tutoriel_btn = QtWidgets.QPushButton("Tutoriel")
        self.generation_btn = QtWidgets.QPushButton("Génération")
        self.quit_btn = QtWidgets.QPushButton("Quitter")

        for btn in [self.tutoriel_btn, self.generation_btn, self.quit_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    border: 2px solid #fff;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 16px;
                    color: white;
                    background-color: #444;
                }
                QPushButton:hover {
                    background-color: #666;  /* Ajout du survol pour highlight */
                }
            """)
            menu_layout.addWidget(btn)

        menu_layout.addStretch(2)
        overlay_layout.addWidget(self.menu_widget, 0)

        self.content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.title_label = QtWidgets.QLabel("Peacekeeper")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-size: 60px; font-weight: bold;")
        content_layout.addStretch(1)
        content_layout.addWidget(self.title_label)
        content_layout.addStretch(2)
        overlay_layout.addWidget(self.content_widget, 1)

        stacked.addWidget(overlay)
        container = QtWidgets.QWidget()
        container.setLayout(stacked)
        self.main_layout.addWidget(container)

        self.quit_btn.clicked.connect(self.show_quit_confirmation)

    def show_quit_confirmation(self):
        dialog = QuitConfirmationDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            QtCore.QCoreApplication.quit()


##############################################################################
# 5a) Gestion de la sortie de l'application
##############################################################################
class QuitConfirmationDialog(QtWidgets.QDialog):
    """Boîte de dialogue de confirmation pour quitter l'application"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setModal(True)
        self.setFixedSize(600, 300)
        self.setStyleSheet("background-color: #000000; border: 5px solid #f55;")
        self.setAutoFillBackground(True)

        # Layout principal
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Message de confirmation
        self.text_label = QtWidgets.QLabel("Voulez-vous vraiment quitter ?")
        self.text_label.setStyleSheet("color: white; font-size: 24px;")
        self.text_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text_label)

        # Boutons
        btn_layout = QtWidgets.QHBoxLayout()

        # Bouton Non
        no_btn = QtWidgets.QPushButton("Non")
        no_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #fff;
                padding: 10px;
                font-size: 16px;
                color: white;
                background-color: #444;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        no_btn.clicked.connect(self.reject)
        btn_layout.addWidget(no_btn)

        # Bouton Oui
        yes_btn = QtWidgets.QPushButton("Oui")
        yes_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #f55;
                padding: 10px;
                font-size: 16px;
                color: white;
                background-color: #944;
            }
            QPushButton:hover {
                background-color: #a55;
            }
        """)
        yes_btn.clicked.connect(self.accept)  # Important: connecté à accept()
        btn_layout.addWidget(yes_btn)

        layout.addLayout(btn_layout)


##############################################################################
# 6) Fenêtre principale, gérant Splash + HomePage dans UN SEUL QStackedWidget
##############################################################################
class MainWindow(QtWidgets.QMainWindow):
    """Fenêtre principale gérant la navigation entre pages.

    Points clés :
    Architecture des couches :
        Utilisation de QStackedLayout avec mode StackAll
        Fond défilant en arrière-plan
        Éléments d'interface en superposition

    Menu latéral :
        Largeur fixe de 220px
        Fond semi-transparent (rgba)
        Boutons stylisés avec effets hover

    Titre central :
        Police [] 60pt en gras
        Ombre portée pour meilleure lisibilité
        Positionnement flexible avec stretches

    Gestion des espaces :
        Marges et espacements précis
        Utilisation de stretch pour centrer
        Suppression des marges inutiles
    """

    def __init__(self):
        super().__init__()

        # Fenêtre plein écran, sans décorations
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        # Splash
        self.splash_page = SplashPage()
        self.splash_page.fade_out_done.connect(self.go_home)
        self.stack.addWidget(self.splash_page)

        # Home
        self.home_page = HomePage(parent=self)
        self.stack.addWidget(self.home_page)

        # On commence sur le splash
        self.stack.setCurrentWidget(self.splash_page)

        # Connecter les boutons
        self.home_page.tutoriel_btn.clicked.connect(self.open_tutoriel)
        self.home_page.generation_btn.clicked.connect(self.open_generation)

    def go_home(self):
        self.stack.setCurrentWidget(self.home_page)

    def open_tutoriel(self):
        sentences = [
            "Bienvenue dans le tutoriel !",
            "Ici, vous apprendrez à utiliser le programme.",
            "Appuyez sur Espace ou A pour passer à la phrase suivante.",
            "Vous pouvez quitter à tout moment en cliquant sur 'Skip tutoriel'."
        ]
        dialog = TutorielDialog(sentences, self)
        dialog.exec()

    def open_generation(self):
        # Ouvre la fenêtre de génération
        try:
            # Chemin absolu depuis la racine du projet
            base_dir = Path(__file__).parent.parent  # nombre de .parent à adapter
            images_folder = base_dir / "Data Bases" / "Celeb A" / "Images" / "selected_images"

            # Vérification du dossier
            if not images_folder.exists():
                raise FileNotFoundError(f"Dossier images introuvable: {images_folder}")

            # Conversion en string pour compatibilité
            folder = str(images_folder)
            dialog = GenerationDialog(folder, self)

        except Exception as e:
            print(f"ERREUR: {e}")
            # Fallback - exemple avec dossier par défaut ou gestion d'erreur
            fallback_folder = str(base_dir / "images_fallback")
            dialog = GenerationDialog(fallback_folder, self)
        dialog.exec()


##############################################################################
# 7) Style de l'application
##############################################################################
class AppStyler:
    """Gestionnaire des styles globaux de l'application"""

    @staticmethod
    def setup_style(app):
        """
        Configure le style visuel global de l'application.

        Parameters
        ----------
        app : QApplication
            Instance de l'application Qt

        Applique :
        - Le style Fusion
        - Une palette de couleurs sombre
        - Les couleurs de base noires
        """
        app.setStyle('Fusion')
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(0, 0, 0))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(255, 255, 255))
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(0, 0, 0))
        app.setPalette(dark_palette)


##############################################################################
# 8) Curseurs personnalisés
##############################################################################
class CursorManager:
    """Gestion centralisée des curseurs personnalisés"""

    @staticmethod
    def get_cursor_paths(relative_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Charge les chemins absolus des curseurs depuis la racine du projet.

        Args:
            relative_path: Chemin relatif personnalisé (optionnel)

        Returns:
            Tuple: (chemin_cursor_resized, chemin_pointer_resized)

        Raises:
            FileNotFoundError: Si le dossier des curseurs est introuvable
        """
        try:
            project_root = Path(__file__).parent.parent
            cursor_dir = project_root / (relative_path or "Elements graphiques/Curseur/dark-red-faceted-crystal-style")

            if not cursor_dir.exists():
                raise FileNotFoundError(f"Dossier curseur introuvable: {cursor_dir}")

            default_path = cursor_dir / "cursor_resized.png"
            pointer_path = cursor_dir / "pointer_resized.png"

            if not default_path.exists():
                logging.warning(f"Fichier curseur par défaut manquant: {default_path}")
            if not pointer_path.exists():
                logging.warning(f"Fichier curseur pointer manquant: {pointer_path}")

            return str(default_path), str(pointer_path)

        except Exception as e:
            logging.error(f"Erreur chargement chemins curseurs: {e}")
            raise

    @staticmethod
    def create(img_path: str, hotspot: Tuple[int, int] = (0, 0)) -> QCursor:
        """
        Crée un curseur personnalisé avec gestion d'erreur améliorée.

        Args:
            img_path: Chemin vers l'image du curseur
            hotspot: Position du point de clic (x,y)

        Returns:
            QCursor: Curseur personnalisé ou curseur par défaut si erreur
        """
        try:
            if not Path(img_path).exists():
                logging.warning(f"Fichier curseur introuvable: {img_path}")
                return QCursor()

            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                logging.error(f"Impossible de charger l'image: {img_path}")
                return QCursor()

            return QCursor(pixmap, *hotspot)

        except Exception as e:
            logging.error(f"Erreur création curseur: {e}")
            return QCursor()

    @staticmethod
    def apply_global_style(app, default_path, pointer_path):
        """
        Applique les styles de curseur globaux.

        Parameters
        ----------
        app : QApplication
            Instance de l'application Qt
        default_path : str
            Chemin vers l'image du curseur par défaut
        pointer_path : str
            Chemin vers l'image du curseur de type pointeur

        Configure :
        - Le curseur par défaut pour toute l'application
        - Le curseur spécial pour les boutons
        - Gère automatiquement la conversion des chemins pour CSS
        """
        default_css = default_path.replace("\\", "/")
        pointer_css = pointer_path.replace("\\", "/")

        app.setStyleSheet(f"""
            * {{
                cursor: url("{default_css}") 15 15, default;
            }}
            QPushButton, QPushButton:hover {{
                cursor: url("{pointer_css}") 15 15, pointer;
            }}
        """)
        app.setOverrideCursor(CursorManager.create(default_path))

    @staticmethod
    def apply_to_hierarchy(widget, cursor):
        """
        Applique un curseur à un widget et toute sa hiérarchie.

        Parameters
        ----------
        widget : QWidget
            Widget racine à partir duquel appliquer
        cursor : QCursor
            Curseur à appliquer

        Parcourt récursivement toute l'arborescence des widgets
        """
        widget.setCursor(cursor)
        for child in widget.findChildren(QtWidgets.QWidget):
            child.setCursor(cursor)


##############################################################################
# 9) Fonctions de vérifications au lacement
##############################################################################
def dezip_images():
    """
    Vérifie si le dossier 'selected_images' contient des fichiers, sinon extrait l'archive zip correspondante.

    Cette fonction s'assure que :
    - Si le dossier 'selected_images' est vide ou inexistant, il est créé.
    - Si une archive 'selected_images.zip' est présente, elle est extraite dans le répertoire parent.
    - Si l'archive zip est absente et que le dossier est vide/inexistant, une exception est levée.

    Notes
    -----
    - Le dossier des images est situé dans :
      ``Data Bases/Celeb A/Images/selected_images``
    - L'archive zip est située dans :
      ``Data Bases/Celeb A/Images/selected_images.zip``

    Raises
    ------
    FileNotFoundError
        Si l'archive zip est absente et que le dossier des images est inexistant ou vide.
    """
    # Définition des chemins absolus
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data Bases", "Celeb A", "Images"))
    folder = os.path.join(base_dir, "selected_images")
    zip_path = os.path.join(base_dir, "selected_images.zip")

    # Vérifier si le dossier existe et contient au moins un fichier
    if os.path.exists(folder) and any(os.scandir(folder)):
        return  # Dossier déjà rempli, pas besoin de dézipper

    # Si le dossier est vide ou inexistant, tenter de dézipper l'archive
    if os.path.exists(zip_path):
        os.makedirs(folder, exist_ok=True)  # Créer le dossier s'il n'existe pas
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)  # Extraire dans le dossier parent pour recréer la structure
    else:
        raise FileNotFoundError(f"L'archive zip n'a pas été trouvée : {zip_path}")


##############################################################################
# 10) Lancement
##############################################################################
def main():
    """Point d'entrée principal de l'application."""
    dezip_images()
    app = QtWidgets.QApplication(sys.argv)
    AppStyler.setup_style(app)

    try:
        # Chargement des chemins avec gestion d'erreur
        default_path, pointer_path = CursorManager.get_cursor_paths()

        # Application des curseurs
        CursorManager.apply_global_style(app, default_path, pointer_path)
        window = MainWindow()
        CursorManager.apply_to_hierarchy(window, CursorManager.create(default_path))

    except FileNotFoundError as e:
        logging.error(str(e))
        # Mode dégradé avec curseurs système
        window = MainWindow()
    except Exception as e:
        logging.critical(f"Erreur critique initialisation: {e}")
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
