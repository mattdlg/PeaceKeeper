from setuptools import setup, find_packages

setup(
    name='projet_4bim_test_1_agnc',
    version='0.1.0',
    author='DELEGLISE Matthieu, DURAND Julie, BEL MELIH Morad, FREMAUX Philippine, ANIBOU Amrou',
    author_email='amrou.anibou@insa-lyon.fr',
    description='Test 1 déploiement projet 4bim',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://gitlab.insa-lyon.fr/mdeleglise/projet-4bim',
    packages=find_packages(include=['AlgoGenetique', 'AlgoGenetique.*', 'Autoencodeur', 'Autoencodeur.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
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
    ],
)
