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
    Create an array containing a given number of 1D vectors
    all having the same size, whose coordinates are in the 
    interval [0, 1].

    Parameters
    ----------
    size : int
        Size of the vectors 
    nb_to_create : int
        Number of vectors to generate in the array

    Returns 
    -------
    vectors : np.array
        Array containing the generated 1D vectors.

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
    Create an array containing a given number of 1D vectors
    all having the same size, and whose coordinates are
    only 0 or 1.

    Parameters
    ----------
    size : int
        Size of the vectors 
    nb_to_create : int
        Number of vectors to generate in the array

    Returns 
    -------
    vectors : np.array
        Array containing the generated 1D black and white vectors.

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
    Linear interpolation of some vectors between the two 
    input vectors.

    Parameters
    ----------
    start_vector : np.array
        Lower bound vector of the linear interpolation interval
    end_vector : np.array
        Upper bound vector of the linear interpolation interval
    nb_to_create : int
        Number of vectors to generate in the array

    Returns 
    -------
    vectors : np.array
        Array containing the interpolated 1D  vectors.

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

if __name__ == "__main__":
    np.random.seed(42)
    # list_vectors = create_random_vectors(4, 3)
    list_vectors = create_black_and_white_vectors(10,3)
    print(list_vectors)


    fst  = np.array([4, 4, 1, 3, 1, 4, 3, 2, 5, 2])
    snd = np.array([1, 1, 3, 4, 1, 5, 5, 5, 4, 3])
    list_vectors = interpolate_vectors(fst, snd, 5)
    print(list_vectors)

