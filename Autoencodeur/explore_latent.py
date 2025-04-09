"""
Projet Portrait robot numérique 
-------------------------------
Exploration :
    Création de vecteurs appartenant à l'espace latent
    d'un autoencodeur et affichage des images décodées.
    Cette algorithme vise à explorer les coordonnées de 
    l'espace latent afin de mieux comprendre sa structure.
-------------------------------
Auteurs : 
    Deléglise Matthieu 
-------------------------------
Version : 
    1.1 (07/04/2025)
"""

#### Libraries ####
import numpy as np

#### Creation of vectors ####
def create_random_vectors(size, nb_to_create):
    """
    Créé un array contenant un nombre donné de vecteurs 1D
    tous de la même taille, dont les coordonnées sont tous 
    dans l'intervalle [0, 1].

    Paramètres
    ----------
    size : int
        Taille des vecteurs
    nb_to_create : int
        Nombre de vecteurs à générer

    Retours 
    -------
    vectors : np.array
        Tableau contenant les vecteurs 1D générés.
        Chaque ligne correspond à un vecteur.

    >>> np.random.seed(42)
    >>> create_random_vectors(4, 3)
    [[0.37454012 0.95071431 0.73199394 0.59865848]
     [0.15601864 0.15599452 0.05808361 0.86617615]
     [0.60111501 0.70807258 0.02058449 0.96990985]]

    """
    vectors = np.random.random_sample((nb_to_create, size))
    return vectors

def create_black_and_white_vectors(size, nb_to_create):
    """
    Créé un tableau contenant un nombre donné de vecteurs 1D
    ayant tous la même taille et dont les coordonnées sont
    uniquement 0 ou 1.

    Paramètres
    ----------
    size : int
        Taille des vecteurs
    nb_to_create : int
        Nombre de vecteurs à générer

    Retours 
    -------
    vectors : np.array
        Tableau contenant les vecteurs "noir et blanc" générés.
        Chaque ligne correspond à un vecteur.

    >>> np.random.seed(42)
    >>> create_black_and_white_vectors(10,3)
    [[0 1 0 0 0 1 0 0 0 1]
     [0 0 0 0 1 0 1 1 1 0]
     [1 0 1 1 1 1 1 1 1 1]]

    """
    vectors = np.random.randint(0, 2, (nb_to_create, size))
    return vectors

def interpolate_vectors(start_vector, end_vector, nb_to_create):
    """
    Interpolation linéaire de certains vecteurs entre 
    les deux vecteurs en entrées.

    Cette fonction à pour but de générer une succession d'image
    qui font la transition entre les deux images dont les vecteurs 
    dans l'espace latent sont donnés en entrée.

    Paramètres
    ----------
    start_vector : np.array
        Vecteur limite basse de l'interval d'interpolation linéaire
    end_vector : np.array
        Vecteur limite haute de l'interval d'interpolation linéaire
    nb_to_create : int
        Nombre de vecteurs à générer

    Retours 
    -------
    vectors : np.array
        Tableau contenant les vecteurs interpolés.
        Chaque ligne correspond à un vecteur.

    >>> fst  = np.array([4, 4, 1, 3, 1, 4, 3, 2, 5, 2])
    >>> snd = np.array([1, 1, 3, 4, 1, 5, 5, 5, 4, 3])
    >>> interpolate_vectors(fst, snd, 5)
    [[4.   4.   1.   3.   1.   4.   3.   2.   5.   2.  ]
     [3.25 3.25 1.5  3.25 1.   4.25 3.5  2.75 4.75 2.25]
     [2.5  2.5  2.   3.5  1.   4.5  4.   3.5  4.5  2.5 ]
     [1.75 1.75 2.5  3.75 1.   4.75 4.5  4.25 4.25 2.75]
     [1.   1.   3.   4.   1.   5.   5.   5.   4.   3.  ]]

    """
    fraction = np.linspace(0, 1, nb_to_create)
    vectors = np.zeros((nb_to_create, start_vector.shape[0]))
    for i in range(nb_to_create) :
        vectors[i] = start_vector + (end_vector - start_vector) * fraction[i]
    return vectors

def explore_one_coord(coord_index, increment, size, nb_to_create):
    """
    Créer un tableau contenant un nombre donné de vecteurs 1D, 
    tous de même taille et dont les coordonnées sont toutes nulles, 
    sauf celles dont l'index est donné en paramètre, augmentées 
    d'une valeur arbitraire dans chaque vecteur du tableau par rapport au précédent.

    Cette fonction à pour but de repérer ce qu'un petit groupe de coordonnées
    de l'espace latent encode dans l'image d'origine.

    Paramètres
    ----------
    coord_index : int ou list
        Index des coordonnées à modifier dans chaque vecteur.
        Chaque index doit appartenir à [0, size[.
    increment : float
        Valeur pour incrémenter les coordonnées à modifier.
    size : int
        Taille des vectors.
    nb_to_create : int
        Nombre de vecteurs à générer.

    Retours 
    -------
    vectors : np.array
        Tableau contenant les vecteurs 1D générés.

    >>> explore_one_coord(5,10,3)
    [[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.   0.25 0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.   0.5  0.   0.   0.   0.  ]]

    >>> explore_one_coord([2,5,8],1,10,3)
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 1. 0. 0. 1. 0.]
     [0. 0. 2. 0. 0. 2. 0. 0. 2. 0.]]

    """
    vectors = np.zeros((nb_to_create, size))
    for i in range(1, nb_to_create):
        vectors[i][coord_index] = vectors[i-1][coord_index] + increment
    return vectors

if __name__ == "__main__":

    np.random.seed(42)
    
    #### Test des fonctions ####
    list_vectors = create_random_vectors(4, 3)
    print(list_vectors)

    list_vectors = create_black_and_white_vectors(10,3)
    print(list_vectors)


    fst  = np.array([4, 4, 1, 3, 1, 4, 3, 2, 5, 2])
    snd = np.array([1, 1, 3, 4, 1, 5, 5, 5, 4, 3])
    list_vectors = interpolate_vectors(fst, snd, 5)
    print(list_vectors)

    print(explore_one_coord([2,5,8],1,10,3))

