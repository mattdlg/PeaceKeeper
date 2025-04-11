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
from typing import Tuple, Optional
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AlgoGenetique import user_driven_algo_gen as udGA
from utils_autoencoder import load_best_hyperparameters, Autoencoder, device, transform_

best_params = load_best_hyperparameters("best_hyperparameters.pth")


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

    This class allows viewing of images, user selection and reconstruction
    with the genetic algorithm when the user clicks "Generation".
    """

    def __init__(self, image_folder, parent=None):
        """
        Creation of an instance of the GenerationDialog class.

        Parameters
        ----------
        image_folder : str
            Path to the images directory.
        parent : QWidget, optional
            Parent widget (default is None).

         Attributes
        ----------
        image_folder : str
            Path to the images directory.
        all_image_paths : list
            Liste contenant les chemins d'accès des fichiers présents dans le dossier Images.
        autoencoder : AutoencoderModel
            Instance used for image encoding and decoding.
        selected_images : list
            Stores paths of images selected by the user.
        selected_buttons : list
            Stores references to the clicked QPushButton instances.
        visualized_images : set
            Images already shown to the user.
        button_image_map : dict
            Links each button to its corresponding image.
        main_layout : QVBoxLayout
            The main vertical layout of the dialog window.
        initial_section_layout : QVBoxLayout
            Layout containing 10 clickable image buttons.
        original_layout : QVBoxLayout
            Layout used to display the two selected images buttons.
        reconstructed_section_layout : QVBoxLayout
            Layout used to display 6 new portraits with variations.
        random_img_section_layout : QVBoxLayout
            Layout used to display 4 randoms images.
        final_image_layout : QHBoxLayout
            Layout used to display the final portrait.
        button_layout : QHBoxLayout
            Layout for action buttons such as "Validate", "New Images", "Final Portrait", and "Close".
        self.transforms : torchvision.transforms.Compose
            A composition of preprocessing transformations applied to input images before
            passing them through the model. Includes resizing, center cropping, and normalization
            to tensor format.
        self.device : torch.device
            Specifies the device on which tensors and the autoencoder model will be allocated.
            Automatically selects 'cuda' if a GPU is available; otherwise defaults to 'cpu'.
        self.model : Autoencoder
            The convolutional autoencoder model used to encode and decode images.
            It is initialized with the best hyperparameters.
        """
        super().__init__(parent)
        self.setWindowTitle("Variation des portraits")

        self.image_folder = image_folder
        self.all_image_paths = glob.glob(os.path.join(self.image_folder, "*"))
        self.selected_images = []  # Liste pour garder trace des images sélectionnées
        self.selected_buttons = []
        self.visualized_images = set()  # Ensemble pour garder trace des images déjà affichées
        self.button_image_map = {}  # Dictionnaire pour lier les boutons aux images

        # Configuration of the window
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setModal(True)
        self.setStyleSheet("background-color: #000000; border: none;")
        self.setAutoFillBackground(True)
        self.showFullScreen()
        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

        # Principal Layout
        #self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        print("Layout principal initialisé avec succès !")

        # Create a layout to display 10 images
        self.initial_section_layout = QtWidgets.QVBoxLayout()
        black_block = QtWidgets.QWidget()
        black_block.setStyleSheet("background-color: black;")
        black_block.setFixedHeight(80)  # Hauteur du bloc noir, ajustable selon le besoin
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
        # self.main_layout.addStretch(1)  # Ajoute un espace flexible entre les deux layouts

        # Layout to display selected images
        self.original_layout = QtWidgets.QVBoxLayout()
        self.title_label = QtWidgets.QLabel("Images sélectionnées")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.title_label.setFixedHeight(40)
        self.original_layout.addWidget(self.title_label)
        self.images_layout = QtWidgets.QGridLayout()
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        #self.images_layout.setSpacing(150)
        self.original_layout.addLayout(self.images_layout)

        # Layout to display images generated by genetic algorithm
        self.reconstructed_section_layout = QtWidgets.QVBoxLayout()
        #self.reconstructed_section_layout.setSpacing(5)
        self.reconstructed_section_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label_choice = QtWidgets.QLabel("Veuillez sélectionner 2 portraits")
        self.title_label_choice.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label_choice.setStyleSheet(
            "color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.reconstructed_section_layout.addWidget(self.title_label_choice)
        self.reconstructed_layout = QtWidgets.QGridLayout()
        self.reconstructed_layout.setContentsMargins(0, 20, 0, 20)
        self.reconstructed_section_layout.addLayout(self.reconstructed_layout)

        # Créer un layout vertical pour les 2 premiers
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.addLayout(self.original_layout)
        #self.left_layout.addStretch(1)
        self.left_layout.addLayout(self.reconstructed_section_layout)

        # Layout to display new randoms images
        self.random_img_section_layout = QtWidgets.QVBoxLayout()
        #self.random_img_section_layout.setSpacing(5)
        self.random_img_section_layout.setContentsMargins(0, 0, 0, 0)
        #self.random_img_section_layout.setFixedWidth(300)
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
        #self.all_images_as_btn_layout.addStretch(1)
        self.all_images_as_btn_layout.addLayout(self.random_img_section_layout)

        # Layout to dispaly definitive portrait
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

        # Layout for buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.button_layout.setSpacing(10)
        self.button_layout.setContentsMargins(0, 0, 0, 0)

        # Add 'Fermer' button
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

        # Add 'Valider' button
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

        # Add "Nouvelles images" button
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

        # Add "Portrait définitif" button
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
        self.model.load_state_dict(torch.load('conv_autoencoder.pth', map_location=self.device))
        self.transforms = transform_

        # Charger les images
        self.load_images()

    def reset_selected_buttons(self):
        """
        Réinitialise la liste des boutons et images sélctionnés et retire l'effet
        d'ombrage de ces images pour prévenir qu'elles sont déselctionnées.
        Cette fonction est appelée si des images ont été sélectionnées et que l'utilisateur
        décide ensuite de  générer 10 nouvelles images.

        Makes use of
        ------------
        selected_images : list
            Stores paths of images selected by the user.
        selected_buttons : list
            Stores references to the clicked QPushButton instances.

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
        Selects up to 10 images that have not yet been visualized from the specified folder.
        Each image is displayed as a button using method 'display_buttons'.
        Once an image is loaded, it is marked as visualized.
        If no unviewed images remain, a warning message box is displayed to the user.

        Images are resized for consistency and buttons are added to a 5x2 grid. Each button allows
        the user to select images for further interaction.

        Parameters
        ----------
        None

        Makes use of
        ------------
        self.selected_buttons : list
            Stores references to the clicked QPushButton instances.
        self.image_folder : str
            Path to the images directory.
        self.visualized_images : set
            Images already shown to the user.
        self.initial_layout : QtWidgets.QGridLayout
            Layout in which the image buttons are added.
        self.display_buttons : method
            Method used to add images as buttons to a given layout.
        self.reset_selected_buttons : method
            Réinitialisation de la liste des images et retire l'effet d'ombrage de ces images.

        Returns
        -------
        None
        """
        print("Début du chargement des images...")

        # Récupérer les chemins des images
        # image_paths = glob.glob(os.path.join(self.image_folder, "*"))[:10]
        # Récupérer les chemins des images qui n'ont pas encore été visualisées
        #image_paths = [img_path for img_path in glob.glob(os.path.join(self.image_folder, "*"))if img_path not in self.visualized_images][:10]
        # Récupérer les chemins des images qui n'ont pas encore été visualisées
        #all_image_paths = glob.glob(os.path.join(self.image_folder, "*"))

        if self.selected_buttons:
            self.reset_selected_buttons()

        remaining_images = [img_path for img_path in self.all_image_paths if img_path not in self.visualized_images]

        if not remaining_images:
            print("Aucune image trouvée !")
            # QtWidgets.QMessageBox.warning(self, "Avertissement", "Impossible de générer de nouvelles images, toutes les images ont été visualisées")
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

        # Selects 10 random images from those not viewed (or less if fewer than 10 remaining)
        image_paths = random.sample(remaining_images, min(10, len(remaining_images)))

        # Add images as a button in the grid
        for i, img_path in enumerate(image_paths):
            self.visualized_images.add(img_path)

        self.display_buttons(self.initial_layout, image_paths, 5, 150)

    def display_original_images(self):
        """
        Display the 2 selected images in the layout.

        Clears the current content of the layout used for displaying previous selected images
        (only images, not the label "Images sélectionnées"),
        then shows the images selected by the user. The layout is updated
        with those images, organized in a 2-column format using the method 'display_buttons'.

        Parameters
        ----------
        None

        Makes use of
        ------------
        self.selected_images : list
            Stores references to the clicked QPushButton instances.
        self.images_layout : QtWidgets.QGridLayout
            Layout in which the image buttons are added.
        self.display_buttons : method
            Method used to add images as buttons to a given layout.

        Returns
        -------
        None
        """
        print(f"Images sélectionnées : {self.selected_images}")

        for i in reversed(range(self.images_layout.count())):
            widget = self.images_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.display_buttons(self.images_layout, self.selected_images, 2, 150)

    def display_buttons(self, layout, data_img, nb_col, target_size):
        """
        Display a list of images as clickable buttons in the given layout.

        Handles both image file paths for the first visualisation (as strings, invalid images are replaced with a gray placeholder)
        and NumPy arrays (reconstructed images).
        For each image, a QPushButton is created with the image as its icon. These buttons are arranged
        in the specified layout using a grid format with a defined number of columns. Each button is linked
        to a callback function for image selection and is stored in the 'button_image_map' dictionary
        to maintain its associated image data.

        Parameters
        ----------
        layout : QGridLayout
            The layout in which the image buttons will be added.
        data_img : list
            A list of images to display. Each item can be a file path or a NumPy array.
        nb_col : int
            Number of columns to arrange the image buttons in the layout.
        target_size : int
            Target size of the image buttons in the layout.

        Makes use of
        ------------
        self.button_image_map : dict
            Links each button to its corresponding image.
        self.select_image_from_generated : method
            Method used to update selected images by user.

        Returns
        -------
        None
            This method modifies the layout.
        """

        # Add images as buttons
        for i, img_array in enumerate(data_img):
            if isinstance(img_array, str):
                img_qpixmap = QtGui.QPixmap(img_array)
                if img_qpixmap.isNull():
                    img_qpixmap = QtGui.QPixmap(target_size, target_size)  #targe_size à la place
                    img_qpixmap.fill(QtGui.QColor("gray"))
                #img_qpixmap = img_qpixmap.scaled(target_size, target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            elif isinstance(img_array, np.ndarray):
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                #img_pil = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
                img_qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(
                    img_pil.tobytes("raw", "RGB"),
                    img_pil.width, img_pil.height, img_pil.width * 3, QtGui.QImage.Format.Format_RGB888
                ))
            img_qpixmap = img_qpixmap.scaled(target_size, target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
            img_width = img_qpixmap.width()
            img_height = img_qpixmap.height()

            # Creation of button
            btn = QtWidgets.QPushButton(self)
            btn.setFixedSize(img_width, img_height)
            btn.setIcon(QtGui.QIcon(img_qpixmap))
            btn.setIconSize(QtCore.QSize(img_width, img_height))
            btn.setStyleSheet("border: none;")
            btn.clicked.connect(lambda checked, p=img_array, b=btn: self.select_image_from_generated(b))
            # btn.clicked.connect(lambda checked: self.on_button_click(checked, img_array, btn))
            # btn.clicked.connect(lambda checked, p=img_array, b=btn: self.say_hello(p, b))

            # Associate the button to its numpy in the dictionary
            self.button_image_map[btn] = img_array

            layout.addWidget(btn, i // nb_col, i % nb_col)

    def display_images(self, layout, img_size):
        """
        Display selected images in the specified layout.
        This method iterates over 'self.selected_images', which can contain either
        file paths (str) or NumPy arrays representing images.

        Parameters
        ----------
        layout : QGridLayout
            The layout where the images will be displayed.
        img_size : int
            The target size (img_size=width=height) for the displayed images.

        Makes use of
        ------------
        self.selected_images : list
            Stores paths of images selected by the user.

        Returns
        -------
        None
            The method modifies the given layout by adding images.
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
        This method is called after running the genetic algorithm to generate new
        image variations. It first removes previously displayed images QPushButtons.
        Then, it displays the new images using the 'display_buttons' method in a 3-column grid layout.

        Parameters
        ----------
        generated_images : list
            A list of images with variations from the selected images

        Makes use of
        ------------
        self.reconstructed_layout : QGridLayout
            Layout where the new generated images will be displayed.
        self.validate_btn : QPushButton
            The validation button to sumbmit user choice.
        self.display_buttons : method
            Method used to add images as buttons to the reconstructed layout.

        Returns
        -------
        None
            The method modifies the reconstructed layout .
        """

        # Nettoyer l'affichage actuel
        # Suppression des boutons sauf "Valider", "Fermer"
        for i in reversed(range(self.reconstructed_layout.count())):
            widget = self.reconstructed_layout.itemAt(i).widget()
            if widget and isinstance(widget, QtWidgets.QPushButton):
                # A tester mais je pense qu'on peut enlever
                if widget not in [self.validate_btn]:
                    widget.deleteLater()
                    # print("Bouton supprimé dans reconstructed_layout")

        self.display_buttons(self.reconstructed_layout, generated_images, 3, 150)

    def display_new_randoms_images(self):
        """
        Affiche 4 images randoms encore jamais vu par l'utilisateur.

        Nettoie le contenu actuel du layout (uniquement les précédentest images).
        Ces images sont des boutons cliquables.
        Le layout est mis à jour avec 3 images organisées en format colonne,
        en utilisant la méthode 'display_buttons'.

        Parameters
        ----------
        None

        Makes use of
        ------------
        self.random_img_grid_layout : list
            Layout in which the image buttons are added.
        self.display_buttons : method
            Method used to add images as buttons to a given layout.
        self.all_image_paths : list
            Liste contenant les chemins d'accès des fichiers présents dans le dossier Images.
        self.visualized_images : set
            Images already shown to the user.

        Returns
        -------
        None
        """
        for i in reversed(range(self.random_img_grid_layout.count())):
            widget = self.random_img_grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        remaining_images = [img_path for img_path in self.all_image_paths if img_path not in self.visualized_images]

        if not remaining_images:
            print("Toutes les img ont été affichées")
            # ajouter qqchose pour afficher une img grise
            return

        image_paths = random.sample(remaining_images, min(3, len(remaining_images)))

        for i, img_path in enumerate(image_paths):
            self.visualized_images.add(img_path)

        self.display_buttons(self.random_img_grid_layout, image_paths, 1, 150)

    def display_definitive_portrait(self):
        """
        Display the definitive portrait based on the selected image.

        This method first checks if exactly one image has been selected. If not, it
        shows a warning message box to ask the user to select exactly one image.
        If one image is selected, it clears the principal layout
        (removes unnecessary buttons) and updates the final reconstruction layout
        to show the final portrait.

        Parameters
        ----------
        None

        Makes use of
        ------------
        self.selected_images : list
            Stores references to the clicked QPushButton instances.
            Length of list should be equal to one here.
        self.selected_buttons : list
            A list of selected buttons, cleared at the beginning of the method.
        self.button_layout : QHBoxLayout
            Layout for action buttons.
        self.main_layout : QVBoxLayout
            The main vertical layout of the dialog window.
        self.final_reconstruction_layout : QVBoxLayout
            The layout used to display the final portrait.
        self.display_images : method
            Displays the images on the layout with a specified size.

        Returns
        -------
        None
            The method modifies the final reconstruction layout and the user interface directly.
        """
        self.selected_buttons.clear()
        print(
            f"Après maj selected_images, len(img) : {len(self.selected_images)} et len(btn) : {len(self.selected_buttons)}")

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
        print("Il y a bien qu'une seule image sélectionnée !")

        self.button_layout.removeWidget(self.validate_btn)
        self.validate_btn.deleteLater()
        self.button_layout.removeWidget(self.final_btn)
        self.final_btn.deleteLater()
        self.main_layout.removeItem(self.button_layout)
        self.remove_layout(self.main_layout, self.original_layout)
        self.remove_layout(self.main_layout, self.reconstructed_section_layout)
        self.remove_layout(self.main_layout, self.random_img_section_layout)
        self.main_layout.addLayout(self.final_reconstruction_layout)
        #self.final_reconstruction_layout.addStretch(1)
        #self.main_layout.addStretch(1)
        self.main_layout.addLayout(self.button_layout)
        self.display_images(self.final_image_layout, 400)

    def generate_new_images(self, selected_images):
        """
        Generate 6 new images from the two selected images using a genetic algorithm.

        This method processes the selected images by first encoding them into latent vectors
        using an autoencoder model. Then, it applies a genetic algorithm to create 6 new images
        based on these encoded vectors. Finally, the new images are decoded (reconstructed).
        Finally, the new images are displayed thanks to 'display_generated_images' method.

        Parameters
        ----------
        selected_images : list
            A list containing paths to the selected images (as strings) or numpy arrays
            representing the selected images. These images will be used to generate new ones.

        Makes use of
        ------------
        self.transforms : torchvision.transforms.Compose
            A composition of preprocessing transformations applied to input images before
            passing them through the model. Includes resizing, center cropping, and normalization
            to tensor format.
        self.device : torch.device
            Specifies the device on which tensors and the autoencoder model will be allocated.
            Automatically selects 'cuda' if a GPU is available; otherwise defaults to 'cpu'.
        self.model : Autoencoder
            The convolutional autoencoder model used to encode and decode images.
            It is initialized with the best hyperparameters.
        self.selected_images : list
            Stores references to the clicked QPushButton instances
            (will be used as input to the genetic algorithm).
        self.button_layout : QHBoxLayout
            Layout for action buttons.
        self.final_btn : QPushButton
            The "Portrait définitif" button, which is displayed once the new images are generated
            to choose the final portrait image.
        self.display_original_images : method
            Displays the images selected by the user.
        self.display_generated_images : method
            Displays the new images generated by the genetic algorithm.

        Returns
        -------
        None
            The method modifies the user interface by displaying both the selected images and
            the generated images.
        """

        # Charger et encoder les images sélectionnées
        list_vectors = []

        """
        for img_path in selected_images:
            img = Image.open(img_path).convert("RGB")
        """

        for img_data in selected_images:
            # Si c'est un chemin de fichier (les 10 images générées initialement):
            if isinstance(img_data, str):
                img = Image.open(img_data).convert("RGB")
                # print("Les images sont converties en RGB")
            # Si c'est une liste de numpy array (les images regénérées),
            # convertir en PIL.Image
            elif isinstance(img_data, np.ndarray):
                # Vérifier si c'est la bonne conversion
                img = Image.fromarray((img_data * 255).astype(np.uint8))
            else:
                print(f"Type de donnée inattendu : {type(img_data)}")
                continue  # On saute cet élément s'il est invalide

            tensor_img = self.transforms(img).unsqueeze(0).to(self.device)
            #tensor_img = self.autoencoder.transforms(img).unsqueeze(0).to(self.autoencoder.device)

            with torch.no_grad():
                latent_vector = self.model.encode(tensor_img)
                list_vectors.append(latent_vector[0].cpu().numpy())

        # Appliquer l'algorithme génétique pour générer 6 nouvelles images
        # new_targets = GAm.create_multiple_target_from_pictures([v[0] for v in list_vectors], 6)
        # solutions = GAm.run_multiple_ga(new_targets)
        solutions = udGA.run_ga(list_vectors, nb_solutions=6, crossover_method="single-coordinate", mutation_rate=0,
                                sigma_mutation=0.2)
        # Convertir en tenseur PyTorch
        sol = torch.tensor(solutions, dtype=torch.float32)
        sol = sol.view(solutions.shape[0], list_vectors[0].shape[0])
        #sol = torch.tensor(solutions, dtype=torch.float32).view(solutions.shape[0], 128, 8, 8)

        with torch.no_grad():
            reconstructed = self.model.decode(sol).cpu().numpy().transpose(0, 2, 3, 1)
        print("Les solutions sont reconstruites")
        # print(f"Les images originales sont : {self.selected_images}")

        if self.button_layout.indexOf(self.final_btn) == -1:  # Vérifie si le bouton est déjà dans le layout
            self.button_layout.addWidget(self.final_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # Display selected images and images generated by GA as buttons
        print(self.selected_images)
        self.display_original_images()
        self.selected_images.clear()
        self.display_generated_images(reconstructed)
        self.display_new_randoms_images()
        print("L'étape de visualisation est réussit")

    def validate_selection(self):
        """
        Validates the selection of two images and displays the reconstructed image
        of the first selected image.

        This method ensures that exactly two images have been selected by the user.
        Once the selection is validated, the main layout is updated by removing
        the initial layout with 10 images and displaying the reconstructed version
        with layout containing selected images and images generated by GA.
        'generate_new_images' is called.

        Parameters
        ----------
        None

        Makes use of
        ------------
        self.selected_images : list
            Stores references to the clicked QPushButton instances.
            Length of list should be equal to two here.
        self.main_layout : QVBoxLayout
            The main layout of the window.
        self.reconstructed_section_layout : QVBoxLayout
            Layout used to display 6 new portraits with variations.
        self.button_layout : QHBoxLayout
            Layout for action buttons, which is updated during the validation process.
        self.load_new_img : QPushButton
            A button to generate 10 new portraits.
            It is removed if it was present.
        self.original_layout : QVBoxLayout
            Layout used to display the two selected images buttons.
        self.generate_new_images : method
            A method used to generate and display images thanks to GA based on the selected ones.

        Returns
        -------
        None
            The method modifies the user interface by displaying the selected reconstructed images.
        """

        self.selected_buttons.clear()
        print(
            f"Après maj selected_images, len(img) : {len(self.selected_images)} et len(btn) : {len(self.selected_buttons)}")
        print(type(self.selected_images[0]))

        if len(self.selected_images) != 2:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Erreur")
            msg_box.setText("Veuillez sélectionner exactement deux images")

            msg_box.setMinimumSize(500, 300)  # Définit une taille minimale pour la boîte de dialogue
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

        # Suppress layout with 10 first images
        self.remove_layout(self.main_layout, self.initial_section_layout)
        # Suppress "Nouvelles images" button
        if self.load_new_img in [self.button_layout.itemAt(i).widget() for i in range(self.button_layout.count())]:
            self.button_layout.removeWidget(self.load_new_img)
            self.load_new_img.deleteLater()
        self.main_layout.removeItem(self.button_layout)
        # Add layout with selected images and images generated by GA
        self.main_layout.addLayout(self.all_images_as_btn_layout)
        #self.main_layout.addStretch(1)
        self.main_layout.addLayout(self.button_layout)
        self.generate_new_images(self.selected_images)
        # self.open_reconstructed_dialog(img_path)

    def select_image(self, img_path, button):
        """
        This method adds an image to the list of selected images when it is clicked,
        and highlights it by applying a shadow effect to the associated button.
        If the button is already selected, it removes the selection and un-highlights the button.

        Parameters
        ----------
        img_path : str
            The file path of the image that is being selected or unselected.

        button : QPushButton
            The button corresponding to the image.

        Makes use of
        ------------
        self.selected_images : list
            A list that stores the file paths of selected images. This list is updated
            when an image is selected or unselected.

        Returns
        -------
        None
            The method directly updating the list of selected images.

        Notes
        -----
        Use of selected_buttons because it is impossible to search for an exact match
        in a list of a np.ndarray (the image directly).
        """
        if img_path not in self.selected_images:
            self.selected_images.append(img_path)
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setColor(QtGui.QColor(180, 180, 180))
            shadow.setOffset(0, 0)
            button.setGraphicsEffect(shadow)
        else:
            self.selected_images.remove(img_path)
            button.setGraphicsEffect(None)  # Retirer l'effet d'ombrage

    def select_image_from_generated(self, button):
        """
        Selects a button (representing an image generated by GA) from the displayed ones
        and highlights it with a shadow effect.

        This method associates an image generated by GA with its corresponding button and
        highlights the button when it selected by applying a shadow effect. If the
        button is already selected, it removes the selection and un-highlights the button.

        Parameters
        ----------
        button : QPushButton
            The button representing the generated image.

        Makes use of
        ------------
        self.selected_buttons : list
            A list of selected buttons.

        self.selected_images : list
            A list that stores the file paths of selected images. This list is updated
            when an image is selected or unselected.

        self.button_image_map : dict
            Links each button to its corresponding image.

        Returns
        -------
        None
            The method directly updating the list of selected images.

        Notes
        -----
        Use of selected_buttons because it is impossible to search for an exact match
        in a list of a np.ndarray (the image directly).
        """
        associated_img = self.button_image_map.get(button)
        if button not in self.selected_buttons:
            self.selected_buttons.append(button)
            print("Bouton ajouté")
            if associated_img is not None and id(associated_img) not in [id(img) for img in self.selected_images]:
                print("{associated_img} est trouvée")
                self.selected_images.append(associated_img)
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setColor(QtGui.QColor(180, 180, 180))
            shadow.setOffset(0, 0)
            button.setGraphicsEffect(shadow)
        else:
            self.selected_buttons.remove(button)
            self.selected_images = [img for img in self.selected_images if id(img) != id(associated_img)]
            button.setGraphicsEffect(None)

    def remove_layout(self, layout_parent, layout):
        """
        Removes a layout and all of its widgets from the given parent layout.

        Parameters
        ----------
        layout_parent : QLayout
            The parent layout from which the target layout will be removed.

        layout : QLayout
            The layout to be removed.

        Returns
        -------
        None
            The method directly modifies the layout structure by removing the specified
            layout and its widgets.
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
            base_dir = Path(__file__).parent.parent  # Adaptez le nombre de .parent selon votre structure
            images_folder = base_dir / "Data bases" / "Celeb A" / "Images" / "selected_images"

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
    def create(img_path: str, hotspot: Tuple[int, int] = (15, 15)) -> QCursor:
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
      ``Data bases/Celeb A/Images/selected_images``
    - L'archive zip est située dans :
      ``Data bases/Celeb A/Images/selected_images.zip``

    Raises
    ------
    FileNotFoundError
        Si l'archive zip est absente et que le dossier des images est inexistant ou vide.
    """
    # Définition des chemins absolus
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data bases", "Celeb A", "Images"))
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
