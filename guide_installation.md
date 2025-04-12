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

3. Installez l'application avec la commande : 

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-YOUR-USERNAME-HERE
```

4. Vous devriez avoir un output de la forme : 

```bash
Collecting example-package-YOUR-USERNAME-HERE
  Downloading https://test-files.pythonhosted.org/packages/.../example_package_YOUR_USERNAME_HERE_0.0.1-py3-none-any.whl
Installing collected packages: example_package_YOUR_USERNAME_HERE
Successfully installed example_package_YOUR_USERNAME_HERE-0.0.1
```