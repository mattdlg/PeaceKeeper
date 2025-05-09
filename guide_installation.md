# Guide d'installation du logiciel


1. Dans un premier temps, dans un terminal, assurez-vous d'avoir la dernière version de pip sur votre ordinateur : 

``` bash
python3 -m pip install --upgrade pip
```

2. Créez un environnement virtuel et activez-le : 

```bash
python3 -m venv env1

source env1/bin/activate
```

Sur windows, la commande pour aciver l'environnement virtuel est : 

```bash
env1\Scripts\activate
```

3. Mettez à jour wheel : 

```bash
pip install --upgrade pip setuptools wheel
```

4. Installez l'application avec la commande : 

```bash
python3 -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple projet-4bim-test-1-agnc==0.1.3
```

5. Vous devriez obtenir un output de la forme : 

```bash
Collecting example-package-YOUR-USERNAME-HERE
  Downloading https://test-files.pythonhosted.org/packages/.../example_package_YOUR_USERNAME_HERE_0.0.1-py3-none-any.whl
Installing collected packages: example_package_YOUR_USERNAME_HERE
Successfully installed example_package_YOUR_USERNAME_HERE-0.0.1
```

6. En cas d'erreur 

Dans le cas où il y a des problèmes de dépendances (problèmes d'installation des packages python requis), vous pouvez utiliser le script install_packages.py pour installer tous les packages requis au bon fonctionnement de l'appli :

```bash
python3 install_packages.py
```

Vous pouvez alors directement utiliser le repository Git pour lancer l'application