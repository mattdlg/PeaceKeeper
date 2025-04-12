# Description des fichiers du GIT repository

Ici, vous trouverez, pour chaque fichier du GIT, une courte description de son utilité pour le projet. Si vous souhaitez obtenir plus d'information sur le fonction d'un fichier en particulier, référez vous directement à sa documentation.

## Liste des fichiers

### Fichiers généraux

- README.md : décrit le projet et sa structure, ainsi que toutes les informations générales sur le projet, l'équipe et le GIT.- guide_utilisation.md : décrit étape par étape comment utiliser l'outils, depuis son lancement jusqu'à la validation d'un portrait robot numérique
- guide_installation.md : décrit étape par étape comment installer l'outils, depuis son téléchargement jusqu'à son lancement.
- LICENSE : fichier décrivant la licence du projet
- interface.png : capture d'écran de l'interface graphique permettant l'utilisation du logiciel. Egalement affiché dans le README.
- MANIFEST.in : document listant les fichiers ne correspondant pas à du code mais nécessaire au bon fonctionnement du package.
- pyproject.toml et setup.py : documents pour la bonne mise en place du package pour le déploiement/téléchargement.

### Elements graphiques

Ce dossier contient tous les éléments graphiques nécessaires au bon affichage de l'interface. Il s'agit donc d'un fichier mp4 pour la vidéo en arrière plan, d'image png pour le curseur, ... 

### Data Bases/Celeb A/Images

Ce dossier contient le fichier zip selected_images.zip. Il s'agit d'un sous ensemble d'images de la base de donnée CelebA spécialement récupéré pour tester le logiciel. Il contient 2600 images. Lors du premier lancement de l'outil (cf guide d'utilisation.md), le zip est automatiquement dézipper dans le dossier Data Bases/Celeb A/Images/selected_images pour que l'interface puisse présenter des images à l'utilisateur.

### ConfirmedSuspects

Ce dossier se remplit au fur et à mesure de l'utilisation du logiciel PeaceKeeper. Il sauvegarde toutes les images ayant été validées en tant que portrait robot définitif lors des précédentes utilisations. Il forme donc une base de données des portraits robots des suscepts.

### AlgoGenetique

- user_driven_algo_gen.py : fichier contenant le code nécessaire au mix des images sélectionnées par l'utilisateur (crossover et mutation). Ce référer à la documentation de ce fichier pour comprendre le fonctionnement de l'algorithme génétique.

### Autoencodeur

- Evolution Modèle : dossier sauvegardant les différentes versions du modèles entrainés sur des lots d'images de plus en plus grand.
- Resultats_exploration_espace_latent : dossier sauvegardant les résultats du code split_latent.py permettant la visualisation de différents vecteurs issus de l'espace latent de l'autoencodeur
- explore_latent.py : code permettant la création de vecteurs spécifiques dans l'espace latent de l'autoencodeur.
- split_latent.py : code permettant de décoder les vecteurs créés dans explore_latent.py et de les afficher pour mieux comprendre la structure de l'espace latent.
- finder_hyperparameters.py : utilise la librairie optuna pour optimiser les paramètres de l'autoencodeur.
- best_hyperparameters.pth : dictionnaire sauvegardant les hyperparamètres trouvés par finder_hyperparameters.py.
- train_autoencoder.py : code permettant l'entrainement du modèle de l'autoencodeur avec les hyperparamètres du dictionnaire hyper_parameters.pth.
- face_autoencoder.db : base de données sauvegarde toutes les MSE testées avec optuna pour tracer les graphes dans plot.py sans avoir a relancer toutes les epochs
- plot.py : affichage des graphiques permettant de visualiser l'adéquation du modèle entrainer (fonction de perte).
- utils_autoencoder.py : reprend les fonctions de train et finder pour être utilisé lors l'encodage/décodage des images pendant l'utilisation de l'outils.
- conv_autoencoder.pth : fichier permettant l'utilisation du modèle entrainé.
