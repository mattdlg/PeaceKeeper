import subprocess
import sys

# Liste des packages nécessaires
required_packages = [
        'Pillow',
        'PyQt6',
        'torch',
        'torchvision',
        'matplotlib',
        'optuna',
        'tqdm',
        'numpy',
        'numba',
        'joblib',
        'scipy',
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} est déjà installé.")
        except ImportError:
            print(f"📦 Installation de {package}...")
            install(package)

if __name__ == "__main__":
    main()