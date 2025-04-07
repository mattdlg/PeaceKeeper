import os
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === 1. Chargement de l'étude depuis la base ===
db_file = None
for f in os.listdir():
    if f.startswith("face_autoencoder") and f.endswith(".db"):
        db_file = f
        break

if db_file is None:
    raise FileNotFoundError("Aucun fichier .db trouvé pour l'étude Optuna.")

study = optuna.load_study(
    study_name="face_autoencoder",
    storage=f"sqlite:///{db_file}"
)

# === 2. Extraction des données des essais ===
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df = df[df["state"] == "COMPLETE"]  # On ne garde que les essais terminés
df = df.rename(columns={"value": "Test MSE"})

print("Colonnes disponibles dans le DataFrame Optuna :")
print(df.columns.tolist())

# === 3. Visualisation de l'effet de chaque hyperparamètre ===
# Adapter les noms de colonnes aux noms utilisés par Optuna
hyperparams = ["params_lr", "params_batch_size", "params_nb_channels", "params_nb_layers", "params_weight_decay"]

for param in hyperparams:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[param], df["Test MSE"])
    plt.title(f"Impact de {"_".join(param.split('_')[1:])} sur la Test MSE")
    plt.xlabel("_".join(param.split('_')[1:]))
    plt.ylabel("Test MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 4. Graphe de convergence (matplotlib) ===
fig = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Historique d'optimisation (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()
