"""
Projet Portrait robot numérique
-------------------------------
Auto-encodeur avec optimisation Optuna des hyperparamètres
-------------------------------
Auteurs :
    Bel Melih Morad, Anibou Amrou et Frémaux Philippine
-------------------------------
Version :
    1.2 (Optimisation Hyperparamètres)
"""

import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_autoencoder import split_data, train_model, CelebADataset, Autoencoder, search_file, data_dir, transform_, device


num_trials = 50 
max_images = 2000  # Nombre d'images à utiliser
train_ratio=0.9

def create_optuna_study(db_filename, reuse_db):
    """
    Crée et retourne une étude Optuna pour l’optimisation d’hyperparamètres et la sauvegarde dans une bd SQLite pour tracer les graphes de convergence
    sans avoir à relancer une nouvelle étude. Une étude Optuna est configurée de sorte à minimiser la MSE en fonction des hyperparamètres. 
    Elle utilise un pruner qui permet d'arrêter prématurément les essais trop mauvais en se basant sur la variance de la MSE des essais précedents. 

    Cette fonction centralise aussi la gestion de la bd Optuna. Si reuse_db est True et que le fichier existe, la bd est supprimée pour en sauvegarder une nouvelle à sa place.
    Sinon l’étude est créée avec un nouveau fichier (voir fonction search_file).


    Paramètres
    ----------
    db_filename : str
        Nom ou chemin vers le fichier de base de données SQLite où l’étude est stockée.
        Exemple : "face_autoencoder.db"

    reuse_db : bool
        - True : conserve la base existante si elle existe, pour continuer l’étude.
        - False : supprime la base existante et recommence une nouvelle étude.

    Retour
    ------
    optuna.study.Study
        Une instance d’étude Optuna prête à être utilisée pour l’optimisation d’hyperparamètres.
    """
    if reuse_db and os.path.exists(db_filename):
        os.remove(db_filename)
        print("Base supprimée.")

    study = optuna.create_study(
        direction='minimize', #Pour indiquer que la MSE doit etre minimisée
        pruner=optuna.pruners.MedianPruner(),
        study_name="face_autoencoder",
        storage=f"sqlite:///{db_filename}",
    )
    return study

def ask_hyperparameters(trial):
    """
    Génère le dictionnaire d’hyperparamètres à tester pour un essai donné.
    C'est là que les plages de recherche pour chaque hyperparamètre sont définies.
    Utilise la méthode suggest_ donnée par la librairie pour parcourir les plages.

    - lr (learning rate) échantillonné logarithmiquement entre 1e-5 et 1e-3.

    - batch_size valeur discrète. Soit 16 soit 32. 
    Un batch trop petit (<16) produit un signal de gradient trop bruité
    Un batch trop grand (>32) peut entraîner une perte de généralisation. 
    En pratique, 16 et 32 sont souvent retrouvées dans des autoencodeurs sur GPU limités.

    - nb_channels, nombre de canaux du dernier bloc convolutif, parmi 64, 128 ou 256.
    Plus nb_channels est élevé, plus le modèle a une grande capacité (bonne reconstruction) mais moins il est compressé. Nos valeurs testées 
    permettent de parcourir le meilleur équilibre reconstruction/compression

    - nb_layers, nombre de couches convolutionnelles compris entre 4 et 6. Impacte directement la profondeur du réseau. Mais dépasser 6 couches
    peut entrainer soit une réduction spatiale trop importante (à chaque couche, division par 2) soit un surcoût.

    - weight_decay : taux de régularisation lasso appliqué par l’optimiseur pour éviter le surapprentissage. Échantillonné log-uniformément 
    entre 1e-6 et 1e-3

    Paramètres
    ----------
    trial : optuna.trial.Trial
        Instance du trial testé

    Retour
    ------
    dict
        Dictionnaire qui contient les hyperparamètres suggérés pour ce trial.
    """
    return {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
        'nb_channels': trial.suggest_categorical('nb_channels', [64, 128, 256]),
        'nb_layers': trial.suggest_int('nb_layers', 4, 6),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }

def prepare_data(train_set, test_set, batch_size):
    """
    DataLoader charge des mini-batches d’images au lieu d’une seule.

    L'utilisation d'un DataLoader pour charger des images par lots (au lieu de une par une) permet d'utiliser
    plus efficacement la mémoire GPU et de paralléliser le calcul

    Les gradients calculés sur un mini-batch sont plus représentatifs que ceux calculés sur une seule image, ce qui
    stabilise l'optimisation

    Exemple de chargement de batches:
    Si le dataset d'entrainement contient 1000 images et batch_size = 32 :
    train_loader va créer 1000 / 32 = 31,25 ≈ 31 batches
    Il va traiter 31 batches de 32 images + 1 batch final avec 8 images
    Après un epoch, toutes les images auront été vues une fois 

    Paramètres
    ----------
    train_set : torch.utils.data.Dataset
        Ensemble d'entraînement
    test_set : torch.utils.data.Dataset
        Ensemble de validation
    batch_size : int
        Taille des batch utilisés pendant l'entraînement. 

    Retours
    -------
    train_loader : torch.utils.data.DataLoader
        Itérateur sur les minibatchs d'entraînement avec shuffle activé.
    test_loader : torch.utils.data.DataLoader
        Itérateur sur les minibatchs de test, sans shuffle.
    """

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def give_param_model(model, train_loader, test_loader, params):
    """
    Retourne un dictionnaire qui contient tous les éléments pour l'entraînement du modèle.
    Elle regroupe le modèle, la fonction de perte, l’optimiseur, les DataLoaders et le nombre d’epochs.

    Structure les données pour le passage d’arguments à la fonction d'entraînement (voir train_model). 

    - model
        Instance d'Autoencoder, c'est le modèle à entraîner.

    - criterion  
        Fonction de perte utilisée pour valider la qualité de la reconstruction. Ici c'est la MSE entre l'image d'entrée et la reconstruite. 

    - optimizer  
        Initialisé avec lr et weight_decay.

    - train_loader, test_loader
        DataLoaders présentés plus tôt. 

    - num_epochs   
        Le nombre d’époques pour garantir une convergence idéale sans causer de surapprentissage. A ajuster selon les modèles.

    Paramètres
    ----------
    model : torch.nn.Module
        Le modèle à entraîner

    train_loader, test_loader

    params : dict
        Dictionnaire qui contient les hyperparamètres. Appelé pour récuperer lr et `weight_decay`.

    Retour
    ------
    dict
        Contient tous les éléments pour entrainer le modèle et appeler train_model.
    """
    return {
        "model": model,
        "criterion": nn.MSELoss(),
        "optimizer": optim.RMSprop(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']),
        "train_loader": train_loader,
        "test_loader": test_loader,
        "num_epochs": 30,
    }

def record_result(study, trial, trial_id, loss, params):
    """
    Enregistre les résultats d’un essai (trial) et affiche les infos pertinentes sur la console. 

    study.tell(trial, loss) pour signaler à Optuna la valeur de la loss du trial.  
       Optuna utilise un algo d’optimisation bayésienne et se sert de cette valeur pour mettre à jour sa stratégie
       d’échantillonnage.

    Si ce trial est le meilleur observé jusqu’à présent, il est affiché sur la console.  
    
    Paramètres
    ----------
    study : optuna.Study
        Objet Optuna qui contient tous les essais et l'algo d’optimisation bayésienne.

    trial : optuna.Trial
        Le trial actuel dont on veut enregistrer la performance.

    trial_id : int
        Nombre du trial actuel pour l’affichage.

    loss : float
        Valeur finale de la loss du trial à transmettre à Optuna.

    params : dict
        Dictionnaire des hyperparamètres utilisés pour ce trial, à afficher en cas de nouveau meilleur score.
    """

    print(f"Trial {trial_id} - Test loss (MSE): {loss:.4f}")

    study.tell(trial, loss)

    if study.best_trial.number == trial.number:
        print("!!! Nouveau meilleur essai !!!")
        print(f"Test loss: {loss:.4f}")
        print(f"Params: {params}\n")

def main():
    # ==== Récupération/Création base de données Optuna ====
    db_filename, reuse_db = search_file("face_autoencoder", ".db")
    study = create_optuna_study(db_filename, reuse_db)

    # === Chargement et préparation du dataset CelebA ===
    dataset = CelebADataset(data_dir, transform_, max_images)

    with tqdm(total=num_trials, desc="Progression globale") as progression_bar_global:
         # === Boucle principale, essais successifs des hyperparamètres ===
         for _ in range(num_trials):
            trial = study.ask()
            trial_id = trial.number + 1

            # === Sélection aléatoire des hyperparamètres dans les plages définies ===
            params = ask_hyperparameters(trial)

            # === Création du modèle Autoencoder avec ces paramètres ===
            model = Autoencoder(nb_channels=params['nb_channels'], nb_layers=params['nb_layers']).to(device)

            # === Découpage des données en train / test ===
            train_set, test_set = split_data(dataset, train_ratio)
            train_loader, test_loader = prepare_data(train_set, test_set, params['batch_size'])

            # === Composants du modèle mis en dictionnaire / Arguments qui servent au suivi Optuna
            data_model = give_param_model(model, train_loader, test_loader, params)
            optuna_args = {"trial": trial, "trial_id": trial_id, "show_progress": True}

            try:
                # === Entraînement du modèle avec les hyperparamètres proposés ===
                loss = train_model(data_model, optuna_args)

                # === Enregistrement des résultats de l’essai dans Optuna ===
                record_result(study, trial, trial_id, loss, params)
            
            # === Arrêt précoce si modèle nul ===
            except optuna.TrialPruned:
                study.tell(trial, float('inf'))

            # === Mise à jour de la barre de progression ===
            progression_bar_global.update(1)

    # === Résumé final et sauvegarde dans le fihier 'best_hyperparameters.pth' ===
    print("=== RÉSULTATS FINAUX ===")
    print(f"Meilleure test loss: {study.best_value:.4f}")
    print(study.best_params)
    torch.save(study.best_params, 'best_hyperparameters.pth')

if __name__ == "__main__":
    main()
