import glob
import os
import sys
import random

from PyQt6 import QtCore, QtGui, QtWidgets


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
    """Fenêtre affichant une grille d'images générées

    Points clés :

    Initialisation de la fenêtre : Propriétés modales et sans bordure

    Gestion des images :
        Chargement sécurisé avec fallback sur placeholder
        Redimensionnement intelligent (keep aspect ratio)

    Layout :
        Grille 5 colonnes avec positionnement automatique
        Gestion du cas <10 images

    Bouton Fermer :
        Style personnalisé
        Positionnement précis
    """

    def __init__(self, image_folder, parent=None):
        """
        Initialise la boîte de dialogue de génération d'images.

        Args :
            image_folder (str) : Chemin du dossier contenant les images à afficher
            parent (QWidget, optional) : Widget parent. Par défaut None.

        Configuration de base :
        - Fenêtre modale sans bordure (FramelessWindowHint)
        - Taille fixe 900x600 pixels
        - Style noir avec bordure blanche
        - Layout en grille pour les images
        """
        super().__init__(parent)
        # Configuration de la fenêtre
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setModal(True)  # Rend la fenêtre modale
        self.setFixedSize(900, 600)  # Taille fixe
        self.setStyleSheet("background-color: #000000; border: 5px solid #fff;")
        self.setAutoFillBackground(True)  # Remplissage automatique du fond

        # Initialisation du layout principal en grille
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Marges intérieures
        layout.setSpacing(10)  # Espacement entre les cellules

        # Chargement et affichage des images
        image_paths = glob.glob(os.path.join(image_folder, "*"))  # Récupère tous les fichiers
        count = min(10, len(image_paths))  # Limite à 10 images max

        for i in range(count):
            label = QtWidgets.QLabel()  # Crée un label pour chaque image

            # Charge l'image ou crée un placeholder si échec
            pix = QtGui.QPixmap(image_paths[i])
            if pix.isNull():  # Si l'image est invalide
                pix = QtGui.QPixmap(178, 218)  # Crée une image vide
                pix.fill(QtGui.QColor("gray"))  # Remplit en gris

            # Redimensionnement de l'image avec :
            # - Conservation des proportions
            # - Lissage haute qualité
            pix = pix.scaled(150, 150,
                             QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)

            label.setPixmap(pix)  # Applique l'image au label
            layout.addWidget(label, i // 5, i % 5)  # Positionne dans la grille (5 colonnes)

        # Configuration du bouton Fermer
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.setStyleSheet("""
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
        close_btn.clicked.connect(self.accept)  # Ferme la fenêtre quand cliqué
        # Positionne le bouton en bas à droite (ligne 2, colonne 4)
        layout.addWidget(close_btn, 2, 4, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignRight)


##############################################################################
# 3) Nouveau Background Cyberpunk (Cycle d'images avec fade in/fade out)
##############################################################################
class CyberpunkBackgroundWidget(QtWidgets.QWidget):
    """
    Widget qui affiche cycliquement n images cyberpunk, chacune pendant 10 secondes,
    avec un effet fade in/fade out. Les images sont redimensionnées pour remplir le widget.
    """
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.images = []
        # Charge les images
        # Récupère tous les fichiers jpg du dossier
        all_paths = glob.glob(os.path.join(folder, "*.jpg"))
        if all_paths:
            # Mélange la liste pour un ordre aléatoire
            random.shuffle(all_paths)
            for path in all_paths:
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    self.images.append(pix)
        if not self.images:
            # Fallback si aucune image n'est chargée
            dummy = QtGui.QPixmap(2000, 2000)
            dummy.fill(QtGui.QColor("black"))
            self.images.append(dummy)
        self.current_index = 0

        # QLabel pour afficher l'image
        self.label = QtWidgets.QLabel(self)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.set_current_image()

        # Effet d'opacité sur le label
        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.label)
        self.label.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        # Animation fade-in
        self.fade_in_anim = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in_anim.setDuration(1500)
        self.fade_in_anim.setStartValue(0)
        self.fade_in_anim.setEndValue(1)
        self.fade_in_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

        # Animation fade-out
        self.fade_out_anim = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out_anim.setDuration(1500)
        self.fade_out_anim.setStartValue(1)
        self.fade_out_anim.setEndValue(0)
        self.fade_out_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
        self.fade_out_anim.finished.connect(self.change_image)

        # Timer pour le cycle : chaque 10 secondes
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10000)
        self.timer.timeout.connect(self.start_transition)
        self.timer.start()

        # Démarre le fade-in pour la première image
        self.fade_in_anim.start()

    def set_current_image(self):
        # Redimensionne l'image courante pour remplir le widget en conservant les proportions
        pix = self.images[self.current_index]
        scaled = pix.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                              QtCore.Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)

    def start_transition(self):
        # Lance le fade-out pour passer à l'image suivante
        self.fade_out_anim.start()

    def change_image(self):
        # Passe à l'image suivante
        self.current_index = (self.current_index + 1) % len(self.images)
        self.set_current_image()
        # Lance le fade-in pour la nouvelle image
        self.fade_in_anim.start()

    def resizeEvent(self, event):
        # Lors du redimensionnement, met à jour l'image affichée
        self.set_current_image()
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
    def __init__(self, image_folder, parent=None):
        super().__init__(parent)
        self.image_folder = image_folder
        self.initUI()

    def initUI(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Utilisation du nouveau background Cyberpunk
        # Remplace l'ancien fond défilant par CyberpunkBackgroundWidget
        cyber_folder = os.path.join("Elements graphiques", "Background", "horror_forest_visual_novel_backgrounds_2",
                                    "horror_forest")
        self.background = CyberpunkBackgroundWidget(cyber_folder)

        stacked = QtWidgets.QStackedLayout()
        stacked.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
        stacked.addWidget(self.background)

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
        menu_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                background: qlineargradient(
                    spread:pad, 
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 0, 0, 0),
                    stop:0.2 rgba(30, 30, 30, 80),
                    stop:0.5 rgba(255, 255, 255, 120),
                    stop:0.8 rgba(30, 30, 30, 80),
                    stop:1 rgba(0, 0, 0, 0)
                );
                padding: 8px 0;
                border-radius: 2px;
            }
        """)
        menu_label.setMinimumWidth(180)
        menu_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        menu_layout.addWidget(menu_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        menu_layout.addStretch(1)

        self.tutoriel_btn = QtWidgets.QPushButton("Tutoriel")
        self.tutoriel_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #fff;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
                color: white;
                background-color: #444;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        menu_layout.addWidget(self.tutoriel_btn)

        self.generation_btn = QtWidgets.QPushButton("Génération")
        self.generation_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #fff;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
                color: white;
                background-color: #444;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        menu_layout.addWidget(self.generation_btn)

        menu_layout.addStretch(2)

        self.quit_btn = QtWidgets.QPushButton("Quitter")
        self.quit_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #f55;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
                color: white;
                background-color: #944;
            }
            QPushButton:hover {
                background-color: #a55;
            }
        """)
        self.quit_btn.clicked.connect(self.show_quit_confirmation)
        menu_layout.addWidget(self.quit_btn)

        overlay_layout.addWidget(self.menu_widget, 0)

        self.content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.title_label = QtWidgets.QLabel("Visage X")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_font = QtGui.QFont("Bank Gothic", 60, QtGui.QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: white;")
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.title_label)
        shadow.setBlurRadius(10)
        shadow.setOffset(3, 3)
        shadow.setColor(QtGui.QColor("#000000"))
        self.title_label.setGraphicsEffect(shadow)

        content_layout.addStretch(1)
        content_layout.addWidget(self.title_label)
        content_layout.addStretch(2)
        overlay_layout.addWidget(self.content_widget, 1)

        stacked.addWidget(overlay)
        container = QtWidgets.QWidget()
        container.setLayout(stacked)
        self.main_layout.addWidget(container)

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
    def __init__(self, image_folder):
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
        self.home_page = HomePage(image_folder)
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
        folder = "Data bases/Celeb A/Images/selected_images"  # adapte le chemin
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
# 9) Lancement
##############################################################################


def main():
    """Point d'entrée principal de l'application.
    """
    app = QtWidgets.QApplication(sys.argv)
    AppStyler.setup_style(app)

    cursor_dir = os.path.join("Elements graphiques", "Curseur", "dark-red-faceted-crystal-style")
    default_cursor_path = os.path.abspath(os.path.join(cursor_dir, "cursor_resized.png"))
    pointer_cursor_path = os.path.abspath(os.path.join(cursor_dir, "pointer_resized.png"))

    CursorManager.apply_global_style(app, default_cursor_path, pointer_cursor_path)
    window = MainWindow("Data bases/Celeb A/Images/selected_images")
    CursorManager.apply_to_hierarchy(window, CursorManager.create(default_cursor_path))
    window.show()
    ret = app.exec()
    sys.exit(ret)


if __name__ == "__main__":
    main()
