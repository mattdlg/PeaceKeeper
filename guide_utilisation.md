# Guide d'utilisation du logiciel

Si vous n'avez pas encore installer le logiciel sur votre machine, assurez vous de consulter le guide d'installation (guide_installation.md) avant toute tentative d'utilisation.

1. Dans un terminal, lancer python à l'aide de la commande :

```bash
$ python ou $ python3
```

2. Importer le code nécessaire à l'exécution du logiciel :

```python
From PeaceKeeper import new_inference
```

3. Lancer la fonction principale de new_inference 

```python
new_inference.main()
```

Une interface graphique permettant la création d'un portrait robot numérique s'ouvre suite à l'appel de cette fonction. Cette interface affiche un menu principal depuis lequel l'utilisateur contrôlera l'application

4. A partir de cette interface, plusieurs actions sont possibles :
- Quitter l'application à l'aide du bouton "quitter"
- Lancer le tutoriel d'utilisation (recommandé lors de la première utilisation pour bien comprendre le fonctionnement de l'application).
- Lancer l'outil de création d'un portrait robot numérique.

5. Une fois l'outil de création lancé, un petit lot d'images provenant de la base de données CelebA sont affichées. Si celles-ci ne correspondent en rien à ce qu'attend l'utilisateur (aucun trait commun avec le suspect), ce dernier peut choisir d'afficher un autre lot d'images. Ces nouvelles images sont entièrement nouvelles, c'est à dire qu'elles n'ont encore jamais été montrées à l'utilisateur au cours de cette session. Cette opération est réalisable à tout instant au cours de l'utilisation de l'outil, et permet donc de repartir de 0.

6. Dans le cas où au moins 1 image intéresse l'utilisateur (traits semblables à ceux du suscept, caractéristiques intéressantes à ajouter au portrait robot, ... ), celui-ci peut cliquer sur exactement deux images et utiliser le bouton "valider" pour valider sa sélection. 
- S'il sélectionne moins/plus de deux images, un message d'erreur apparaitra l'invitant à sélectionner le bon nombre d'images.
- S'il a sélectionné exactement deux images, l'algorithme génétique est lancé avec ces deux images en entrées.

7. Suite à l'exécution de l'algorithme génétique, de nouvelles images sont affichées. Parmi celles-ci, on trouve : 
- les 2 images que l'utilisateur vient de sélectionner ;
- 3 nouvelles images provenant de la base de données CelebA
- 6 images, solutions de l'algorithme génétique, correspondant à un mix des images sélectionnées par l'utilisateur (Ce référer à la documentation de l'algorithme génétique pour plus d'information sur le processus de génération de ces images mélangées).

8. L'utilisateur peut alors répéter le processus en choisissant deux images parmi toutes celles qui sont affichées, et ce même parmi les 2 images qu'il a sélectionné à l'étape précédente.

9. Une fois qu'une des images affichées en solution de l'algorithme génétique correspond à peu près à l'image que ce fait l'utilisateur du suspect, il peut utiliser le bouton "portrait définitif" pour afficher l'image en grand et confirmer le portrait robot final 