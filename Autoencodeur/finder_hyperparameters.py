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
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils_autoencoder import split_data, train_model, CelebADataset, Autoencoder

# ===== Configuration Globale =====
image_size = (128, 128)
data_dir = "img_align_celeba"
max_images = 2000  # Nombre d'images à utiliser

transform_ = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])

# Callback pour afficher les résultats intermédiaires
def print_best_trial(study, trial):
    if study.best_trial.number == trial.number:
        print("!!! Nouveau meilleur essai !!!")
        print(f"test loss: {trial.value:.4f}")
        print(f"Params: {trial.params}\n")

# ===== 4. Lancement de l'Optimisation =====
if __name__ == "__main__":
    
    if os.path.exists("face_autoencoder.db"):
        confirm = input("Supprimer l'ancienne base ? (y/n) :")
        if confirm.lower() == "y":
            os.remove("face_autoencoder.db")
            print("Base supprimée.")
        else:
            print("Base conservée.")

    study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(),
    study_name="face_autoencoder",
    storage="sqlite:///face_autoencoder.db",  # <- ajout de la persistence
    load_if_exists=True                      # <- recharge si déjà existant
    )
    
    dataset = CelebADataset(data_dir, transform_, max_images)

    with tqdm(total=50, desc="Progression globale") as progression_bar_global:
        for _ in range(50):
            trial = study.ask()
            # Identifiant unique pour le trial
            trial_id = trial.number + 1

            # Hyperparamètres à tester
            params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'nb_channels': trial.suggest_categorical('nb_channels', [64, 128, 256]),
            'nb_layers': trial.suggest_int('nb_layers', 4, 6), 
            #'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            #'loss_type': trial.suggest_categorical('loss_type', ['MSE', 'L1'])
            }
            print(f"\n Début du Trial {trial_id} avec params: {params}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Autoencoder(nb_channels=params['nb_channels'], nb_layers=params['nb_layers']).to(device)
            
            
            train_set, test_set = split_data(dataset, train_ratio=0.9)

            data_model = {
            "model": model,
            "criterion": nn.MSELoss(), #if params['loss_type'] == 'MSE' else nn.L1Loss(),
            #"optimizer": getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']),
            "optimizer": optim.RMSprop(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']),
            "train_loader": DataLoader(train_set, batch_size=params['batch_size'], shuffle=True),
            "test_loader": DataLoader(test_set, batch_size=params['batch_size'], shuffle=False),
            "num_epochs": 30, 
            }

            optuna_args={
                        'trial': trial,
                        'trial_id': trial_id,
                        'show_progress': True
            }
            
            try:
                loss = train_model(data_model, optuna_args={
                    'trial': trial,
                    'trial_id': trial_id,
                    'show_progress': True
                })
                
                # # === Pénalisation de la complexité ===
                # alpha = 7e-6
                # latent_penalty = model.latent_dim_size()

                # score = loss + alpha * latent_penalty

                print(f"Trial {trial_id} - Test loss (MSE ou L1): {loss:.4f}")

                study.tell(trial, loss)

                # Affichage si c’est le meilleur
                if study.best_trial.number == trial.number:
                    print("!!! Nouveau meilleur essai !!!")
                    print(f"Test loss: {loss:.4f}")
                    print(f"Params: {trial.params}\n")

            except optuna.TrialPruned:
                study.tell(trial, float('inf'))  # indique que ce trial a échoué

            progression_bar_global.update(1)

    # Résumé final
    print("=== RÉSULTATS FINAUX ===")
    print(f"Meilleure test loss: {study.best_value:.4f}")
    print(study.best_params)
        
    torch.save(study.best_params, 'best_hyperparameters.pth')
