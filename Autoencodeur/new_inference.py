import glob
import os
import sys
import zipfile
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AlgoGenetique')))
import user_driven_algo_gen as udGA

from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets


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
        else:
            event.ignore()


##############################################################################
# 2) Fenêtre de Génération (QDialog) - 10 images
##############################################################################
class GenerationDialog(QtWidgets.QDialog):
    """Fenêtre affichant une grille d'images générées"""

    def __init__(self, image_folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Variation des portraits")
        self.setFixedSize(900, 600)

        self.image_folder = image_folder
        self.autoencoder = AutoencoderModel()
        self.selected_images = []  # Liste pour garder trace des images sélectionnées
        self.selected_buttons = []
        self.visualized_images = set()  # Ensemble pour garder trace des images déjà affichées
        self.button_image_map = {}  # Dictionnaire pour lier les boutons aux images

        # Configuration de la fenêtre
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setModal(True)
        self.setFixedSize(900, 600)
        self.setStyleSheet("background-color: #000000; border: 5px solid #fff;")
        self.setAutoFillBackground(True)

        # Layout principal
        self.main_layout = QtWidgets.QVBoxLayout(self)
        print("Layout principal initialisé avec succès !")

        # Créer un layout pour visualiser les 10 images
        self.initial_section_layout = QtWidgets.QVBoxLayout()
        # Créer un label pour le titre
        self.title_label_choice = QtWidgets.QLabel("Veuillez sélectionner 2 portraits")
        self.title_label_choice.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label_choice.setStyleSheet(
            "color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.title_label_choice.setFixedHeight(40)
        self.initial_section_layout.addWidget(self.title_label_choice)
        self.initial_layout = QtWidgets.QGridLayout()
        self.initial_layout.setContentsMargins(0, 70, 0, 50)
        self.initial_section_layout.addLayout(self.initial_layout)
        self.main_layout.addLayout(self.initial_section_layout)
        # self.main_layout.addStretch(1)  # Ajoute un espace flexible entre les deux layouts

        # Layout pour afficher le label des images sélectionnées
        self.original_layout = QtWidgets.QVBoxLayout()
        self.title_label = QtWidgets.QLabel("Images sélectionnées")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.title_label.setFixedHeight(40)
        self.original_layout.addWidget(self.title_label)
        # Layout pour les images (horizontal)
        self.images_layout = QtWidgets.QGridLayout()
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.images_layout.setSpacing(150)
        self.original_layout.addLayout(self.images_layout)

        # Layout pour afficher les images générées
        self.reconstructed_section_layout = QtWidgets.QVBoxLayout()
        self.reconstructed_section_layout.setSpacing(5)  # Réduit l'espace entre le titre et les images
        self.reconstructed_section_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label_choice = QtWidgets.QLabel("Veuillez sélectionner 2 portraits")
        self.title_label_choice.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label_choice.setStyleSheet(
            "color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.reconstructed_section_layout.addWidget(self.title_label_choice)
        self.reconstructed_layout = QtWidgets.QGridLayout()
        self.reconstructed_layout.setContentsMargins(0, 20, 0, 20)
        self.reconstructed_section_layout.addLayout(self.reconstructed_layout)

        # Layout pour afficher le portrait définitif
        self.final_reconstruction_layout = QtWidgets.QVBoxLayout()
        self.title_label = QtWidgets.QLabel("Portrait définitif")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px; padding: 2px; border: none;")
        self.final_reconstruction_layout.addWidget(self.title_label)
        self.final_reconstruction_layout.addSpacing(50)
        # Layout pour les images (horizontal)
        self.final_image_layout = QtWidgets.QHBoxLayout()
        self.final_reconstruction_layout.addLayout(self.final_image_layout)

        # self.main_layout.addStretch(1)  # Ajoute un espace flexible entre les deux layouts

        # Layout pour les boutons (Bas)
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setSpacing(10)  # Ajustez cet espace selon vos besoins
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Supprime les marges

        # Ajouter un bouton de fermeture à droite
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

        # Bouton "Nouvelles images"
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

        # Ajouter un bouton "Portrait définitif"
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

        # Charger les images
        self.load_images()

    def load_images(self):
        """Chargement des images et ajout au layout"""
        print("Début du chargement des images...")

        # Récupérer les chemins des images
        # image_paths = glob.glob(os.path.join(self.image_folder, "*"))[:10]
        # Récupérer les chemins des images qui n'ont pas encore été visualisées
        #image_paths = [img_path for img_path in glob.glob(os.path.join(self.image_folder, "*"))if img_path not in self.visualized_images][:10]
        # Récupérer les chemins des images qui n'ont pas encore été visualisées
        all_image_paths = glob.glob(os.path.join(self.image_folder, "*"))
        remaining_images = [img_path for img_path in all_image_paths if img_path not in self.visualized_images]

        if not remaining_images:
            print("Aucune image trouvée !")
            # Afficher un message disant qu'il n'y a plus d'images disponibles
            # QtWidgets.QMessageBox.warning(self, "Avertissement", "Impossible de générer de nouvelles images, toutes les images ont été visualisées")
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Avertissement")
            msg_box.setText("Impossible de générer de nouvelles images, toutes les images ont été visualisées")

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
            msg_box.exec()  # Affiche la boîte de message
            return

        # Sélectionner 10 images aléatoires parmi celles non visualisées (ou moins si moins de 10 restantes)
        image_paths = random.sample(remaining_images, min(10, len(remaining_images)))

        # Ajouter chaque image sous forme de boutons dans le layout
        for i, img_path in enumerate(image_paths):
            print(f"Chargement de l'image {i + 1} : {img_path}")
            # Ajouter cette image à l'ensemble des images visualisées
            self.visualized_images.add(img_path)

            # Charger l'image (si l'image est invalide, on met une image grise par défaut)
            pix = QtGui.QPixmap(img_path)
            if pix.isNull():
                pix = QtGui.QPixmap(150, 150)
                pix.fill(QtGui.QColor("gray"))

            # Redimensionner l'image
            pix = pix.scaled(150, 150, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)

            img_width = pix.width()
            img_height = pix.height()

            # Création du bouton
            btn = QtWidgets.QPushButton(self)
            btn.setFixedSize(img_width, img_height)
            btn.setIcon(QtGui.QIcon(pix))
            btn.setIconSize(QtCore.QSize(img_width, img_height))
            btn.setStyleSheet("border: none;")
            # btn.clicked.connect(lambda checked, p=img_path: self.open_reconstructed_dialog(p))
            btn.clicked.connect(lambda checked, p=img_path, b=btn: self.select_image(p, b))

            # Ajouter le bouton à la grille
            self.initial_layout.addWidget(btn, i // 5, i % 5)  # Disposition des images dans une grille 5x2

            print(f"Image {i + 1} ajoutée au layout")

            # A voir si on peut pas le sortir de load pcq pas très beau
            # Bouton "Valider" après la sélection d'images
            # self.validate_btn = QtWidgets.QPushButton("Valider")
            # self.validate_btn.clicked.connect(self.validate_selection)
            # Attention !!!! Il est sur le bouton quitter !!!!
            # self.initial_layout.addWidget(self.validate_btn, 2, 3, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        print("Toutes les images ont été ajoutées au layout avec succès !")

    def display_original_images(self):

        print(f"Images sélectionnées : {self.selected_images}")

        # Nettoyer l'affichage actuel
        # Suppression des label (des 2 images) mais pas le label "Images séléctionnées"

        for i in reversed(range(self.images_layout.count())):
            widget = self.images_layout.itemAt(i).widget()
            if widget :
                widget.deleteLater()
                # print("Label supprimé dans original_layout")

        self.display_buttons(self.images_layout, self.selected_images, 2)

    def display_buttons(self, layout, data_img, nb_col):
        """Affiche des images sous forme de boutons cliquables."""

        target_size = 128

        # Ajouter les nouvelles images sous forme de boutons
        for i, img_array in enumerate(data_img):
            if isinstance(img_array, str):
                img_qpixmap = QtGui.QPixmap(img_array)
                #img_qpixmap = img_qpixmap.scaled(target_size, target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                print("Image str chargée")
            # Si c'est une liste de numpy array (les images regénérées),
            elif isinstance(img_array, np.ndarray):
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                #img_pil = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
                img_qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(
                    img_pil.tobytes("raw", "RGB"),
                    img_pil.width, img_pil.height, img_pil.width * 3, QtGui.QImage.Format.Format_RGB888
                ))
                print("Image np chargée")
            img_qpixmap = img_qpixmap.scaled(target_size, target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
            img_width = img_qpixmap.width()
            img_height = img_qpixmap.height()
            print("Dimensions récupérées")

            # Création du bouton
            btn = QtWidgets.QPushButton(self)
            btn.setFixedSize(img_width, img_height)
            btn.setIcon(QtGui.QIcon(img_qpixmap))
            btn.setIconSize(QtCore.QSize(img_width, img_height))
            btn.setStyleSheet("border: none;")
            print("Img assimilée au btn")
            btn.clicked.connect(lambda checked, p=img_array, b=btn: self.select_image_from_generated(p, b))
            print("Btn connecté à select_image_from_generated")
            # btn.clicked.connect(lambda checked: self.on_button_click(checked, img_array, btn))
            # btn.clicked.connect(lambda checked, p=img_array, b=btn: self.say_hello(p, b))

            # Associer le bouton à son tableau numpy dans le dictionnaire
            self.button_image_map[btn] = img_array
            print("Btn ajouté au dico")

            layout.addWidget(btn, i // nb_col, i % nb_col)
            print("Btn ajouté au layout")

    def display_images(self, layout, img_size):
        for img_data in self.selected_images:
            # Créer un QLabel pour chaque image
            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: none; background: none;")

            if isinstance(img_data, str):
                original_img = QtGui.QPixmap(img_data)
            # Si c'est une liste de numpy array (les images regénérées),
            elif isinstance(img_data, np.ndarray):
                # Convertir numpy array en QImage
                img_pil = Image.fromarray((img_data * 255).astype(np.uint8))
                original_img = QtGui.QPixmap.fromImage(QtGui.QImage(
                    img_pil.tobytes("raw", "RGB"),
                    img_pil.width, img_pil.height, img_pil.width * 3,
                    QtGui.QImage.Format.Format_RGB888
                ))
            label.setPixmap(original_img.scaled(img_size, img_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Centre les images horizontalement
            layout.addWidget(label)

    def display_generated_images(self, generated_images):
        """Affiche les 6 nouvelles images générées après application de l'algorithme génétique."""

        # Nettoyer l'affichage actuel
        # Suppression des boutons sauf "Valider", "Fermer"
        for i in reversed(range(self.reconstructed_layout.count())):
            widget = self.reconstructed_layout.itemAt(i).widget()
            if widget and isinstance(widget, QtWidgets.QPushButton):
                # A tester mais je pense qu'on peut enlever
                if widget not in [self.validate_btn]:
                    widget.deleteLater()
                    # print("Bouton supprimé dans reconstructed_layout")

        self.display_buttons(self.reconstructed_layout, generated_images, 3)
        print("Affichage des images générées terminé.")

    def display_definitive_portrait(self):
        self.selected_buttons.clear()
        print(
            f"Après maj selected_images, len(img) : {len(self.selected_images)} et len(btn) : {len(self.selected_buttons)}")

        if len(self.selected_images) != 1:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Erreur")
            msg_box.setText("Veuillez sélectionner exactement une image")

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
            msg_box.exec()  # Affiche la boîte de message
            return
        print("Il y a bien qu'une seule image sélectionnée !")

        self.button_layout.removeWidget(self.validate_btn)
        self.validate_btn.deleteLater()
        self.button_layout.removeWidget(self.final_btn)
        self.final_btn.deleteLater()
        self.main_layout.removeItem(self.button_layout)
        self.remove_layout(self.main_layout, self.original_layout)
        self.remove_layout(self.main_layout, self.reconstructed_section_layout)
        self.main_layout.addLayout(self.final_reconstruction_layout)
        self.final_reconstruction_layout.addStretch(1)
        self.main_layout.addStretch(1)
        self.main_layout.addLayout(self.button_layout)
        self.display_images(self.final_image_layout, 350)

    def generate_new_images(self, selected_images):
        """Génère 6 nouvelles images à partir des 2 images sélectionnées"""

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

            tensor_img = self.autoencoder.transforms(img).unsqueeze(0).to(self.autoencoder.device)

            with torch.no_grad():
                latent_vector = self.autoencoder.model.encode(tensor_img)
                list_vectors.append(latent_vector.cpu().numpy())

        # Appliquer l'algorithme génétique pour générer 6 nouvelles images
        # new_targets = GAm.create_multiple_target_from_pictures([v[0] for v in list_vectors], 6)
        # solutions = GAm.run_multiple_ga(new_targets)
        solutions = udGA.run_ga(list_vectors, nb_solutions=6, crossover_method="blending", mutation_rate=0.1,sigma_mutation=0.1)
        # Convertir en tenseur PyTorch
        sol = torch.tensor(solutions, dtype=torch.float32).view(solutions.shape[0], 128, 8, 8)

        with torch.no_grad():
            reconstructed = self.autoencoder.model.decode(sol).cpu().numpy().transpose(0, 2, 3, 1)
        print("Les solutions sont reconstruites")
        # print(f"Les images originales sont : {self.selected_images}")

        """
        for i in reversed(range(self.initial_layout.count())):
            widget = self.initial_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()  # Supprime le widget
        """

        if self.button_layout.indexOf(self.final_btn) == -1:  # Vérifie si le bouton est déjà dans le layout
            self.button_layout.addWidget(self.final_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # Afficher les images séléctionnées
        print(self.selected_images)
        self.display_original_images()
        self.selected_images.clear()  # vérifier si c'est nécessaire
        # Afficher les nouvelles images
        self.display_generated_images(reconstructed)
        print("L'étape de visualisation est réussit")

    def validate_selection(self):
        """Valide la sélection et affiche l'image reconstruite de la première image sélectionnée"""
        # Une fois la première regénération réalisée,
        # on capture le btn sélectionnée et non l'image donc
        # ici il faut capturer l'image

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

        # self.initial_section_layout.removeWidget(self.title_label_choice)
        # Supprimer l'ancien layout
        self.remove_layout(self.main_layout, self.initial_section_layout)
        # Supprimer le bouton "Nouvelles images"
        if self.load_new_img in [self.button_layout.itemAt(i).widget() for i in range(self.button_layout.count())]:
            self.button_layout.removeWidget(self.load_new_img)
            self.load_new_img.deleteLater()
        # Supprimer 'button_layout' temporairement
        self.main_layout.removeItem(self.button_layout)
        # Ajouter les nouveaux layouts
        self.main_layout.addLayout(self.original_layout)
        self.main_layout.addStretch(1)
        self.main_layout.addLayout(self.reconstructed_section_layout)
        self.main_layout.addStretch(1)
        # Réajouter les boutons en bas
        self.main_layout.addLayout(self.button_layout)
        # Ouvre la fenêtre d'image reconstruite pour la première image
        self.generate_new_images(self.selected_images)
        # self.open_reconstructed_dialog(img_path)

    def select_image(self, img_path, button):
        """Sélectionne une image et l'encadre en rouge"""
        if img_path not in self.selected_images:
            self.selected_images.append(img_path)
            # Ajout d'une ombre portée
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)  # Intensité de l'effet
            shadow.setColor(QtGui.QColor(180, 180, 180))
            shadow.setOffset(0, 0)  # Pas de décalage, effet centré
            button.setGraphicsEffect(shadow)
        else:
            self.selected_images.remove(img_path)
            button.setGraphicsEffect(None)  # Retirer l'effet d'ombrage

    def select_image_from_generated(self, img_array, button):
        """Sélectionne un bouton (image générée) parmi ceux qui sont affichés."""

        # Vérifier si le bouton est déjà dans la liste des boutons sélectionnés
        # Je ne peux pas aller chercher directement si l'img est dans la liste
        # pcq c'est galère de faire ça avec les 3D numpy donc je passe par le bouton
        # que j'ai associé à l'image dans un dictonnaire donc on va pouvoir aller
        # rechercher l'image associée
        # Sûrement pas la méthode la plus jolie mai sj'ai rien trouvé d'autre
        associated_img = self.button_image_map.get(button)
        if button not in self.selected_buttons:
            self.selected_buttons.append(button)
            print("Bouton ajouté")
            # Ajout de l'image correspondante avec son identifiant
            if associated_img is not None and id(associated_img) not in [id(img) for img in self.selected_images]:
                print("{associated_img} est trouvée")
                self.selected_images.append(associated_img)
            # Ajout d'une ombre portée
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)  # Intensité de l'effet
            shadow.setColor(QtGui.QColor(180, 180, 180))
            shadow.setOffset(0, 0)  # Pas de décalage, effet centré
            button.setGraphicsEffect(shadow)
        else:
            self.selected_buttons.remove(button)
            # Supprimer l'image associée en comparant les identifiants
            self.selected_images = [img for img in self.selected_images if id(img) != id(associated_img)]
            button.setGraphicsEffect(None)

    def remove_layout(self, layout_parent, layout):
        """Supprime un layout et tous ses widgets du layout_parent"""

        if layout is not None:
            # Supprimer tous les widgets à l'intérieur du layout
            while layout.count():
                item = layout.takeAt(0)  # Prend l'élément (widget ou sous-layout)

                if item.widget():
                    item.widget().deleteLater()  # Supprime le widget
                elif item.layout():
                    self.remove_layout(layout, item.layout())  # Supprime récursivement les sous-layouts

            # Retirer le layout du parent et le supprimer
            layout_parent.removeItem(layout)


class AutoencoderModel:
    # Classe pour charger et utiliser l'autoencodeur

    def __init__(self, model_path="conv_autoencoder.pth", device=None):  # "Autoencodeur/conv_autoencoder.pth"
        # self.device = torch.device("cpu")  # Forcer l'exécution sur CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transforms = self.create_transforms()

    def load_model(self, model_path):
        # Charge le modèle
        class ConvAutoencoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
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
                self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.Sigmoid(),
                )

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = ConvAutoencoder().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def create_transforms(self):
        # Transformations pour adapter l'image au modèle
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
        ])



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
        video_path = os.path.join("..", "Elements graphiques", "Background",
                                  "night-walk-cyberpunk-city-pixel-moewalls-com.mp4")
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
        folder = "../Data bases/Celeb A/Images/selected_images"  # adapte le chemin
        dialog = GenerationDialog(folder, self)
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
    def create(img_path, hotspot=(15, 15)):
        """
        Crée un curseur personnalisé à partir d'une image.

        Parameters
        ----------
        img_path : str
            Chemin vers l'image du curseur
        hotspot : tuple, optional
            Position du point de clic (x,y), par défaut (15,15)

        Returns
        -------
        QCursor
            Curseur personnalisé ou curseur par défaut en cas d'erreur

        Gère les cas où :
        - Le fichier image n'existe pas
        - L'image ne peut pas être chargée
        """
        pixmap = QtGui.QPixmap(img_path)
        if pixmap.isNull():
            print(f"ERREUR: Impossible de charger {img_path}")
            return QtGui.QCursor()
        return QtGui.QCursor(pixmap, *hotspot)

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
    """Point d'entrée principal de l'application.
    """
    dezip_images()
    app = QtWidgets.QApplication(sys.argv)
    AppStyler.setup_style(app)

    cursor_dir = os.path.join("..", "Elements graphiques", "Curseur", "dark-red-faceted-crystal-style")
    default_cursor_path = os.path.abspath(os.path.join(cursor_dir, "cursor_resized.png"))
    pointer_cursor_path = os.path.abspath(os.path.join(cursor_dir, "pointer_resized.png"))

    CursorManager.apply_global_style(app, default_cursor_path, pointer_cursor_path)
    window = MainWindow()
    CursorManager.apply_to_hierarchy(window, CursorManager.create(default_cursor_path))
    window.show()
    ret = app.exec()
    sys.exit(ret)


if __name__ == "__main__":
    main()
