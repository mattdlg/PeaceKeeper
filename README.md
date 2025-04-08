# Projet 4BIM-INSA_LYON-Biotech_Bioinfo

## Nom
Portrait Robot Numérique

## Description
Ce projet vise à simplifier la création de portraits robots. 
Pour cela, des images issues de la base de donnée CelebA (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, images de visages de célébrités) sont présentés à un utiliseur (la victime) dans une interface graphique. L'utilisateur peut sélectionner deux images afin de les combiner pour en créer de nouvelles. Cette étape est répétée jusqu'à qu'une des combinaisons soit validé par l'utilisateur comme étant assez proche du portrait robot du suspect.

Un Autoencodeur est utilisé pour encoder les images sélectionnées dans un espace latent (compression des pixels en un vecteur de plus petite taille où chaque coordonnée encode une caractéristique particulère de l'image de base). L'Autoencodeur permet aussi de décoder des vecteurs de l'espace latent en images RGB.

Dans l'objectif de combiner deux images entre elle et de créer de la variabilité, un algorithme génétique est également utilisé pour croiser deux vecteurs entre eux et introduire des mutations dans leurs coordonnées. L'étape de sélection de cette algorithme est directement réalisée à chaque sélection de deux images par l'utilisateur.

## Badges

### Langage de programmation et principaux outils utilisés
![Static Badge](https://img.shields.io/badge/Langage-Python-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Tool-Pytorch-blue?logo=pytorch)
![Static Badge](https://img.shields.io/badge/Tool-Numpy-blue?logo=numpy)


## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Amrou Anibou 
Morad Bel Melih
Matthieu Deléglise
Julie Durand
Philippine Fremaux

Nous remercions le tuteur de ce projet, Monsieur Robin Trombetta, pour son aide et ses conseils tout au long du projet

## License
For open source projects, say how it is licensed.

## Project status
Ce projet a été réalisé en suivant la méthode Agile. Actuellement, il est en fin de deuxième sprint. 
Etant seulement à visé scolaire, il ne sera pas maintenu par la suite. 