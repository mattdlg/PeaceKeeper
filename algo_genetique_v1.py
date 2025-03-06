"""
Projet Portrait robot numérique 
-------------------------------
Algorithme génétique :
    Création d'images similaires à une image 
    de départ à l'aide d'un algorithme simulant
    l'évolution d'une population par sélection 
    naturelle.
-------------------------------
Auteurs : 
    Deléglise Matthieu et Durand Julie
-------------------------------
Version : 
    1.0 (06/03/2025)
"""

class GeneticAlgorithm():
    """
    Class GeneticAlgorithm

    This algorithm aims at creating a set of images similar to a target picture using 
    a genetic algorithm. 
    These images are all representing as vectors of dimension n, which are the 
    representation of the images in the latent space of an autoencoder.

    """
    def __init__(self, target):
        """
        Creation of an instance of the GeneticAlgorithm class.

        Parameters
        ----------
        target : np.array
            Vector of size n representing the target photo in the latent space of the autoencoder. 

        Returns
        -------
        None

        """
        self.target_photo = target

    def create_random_init_pop(self):
        return
    
    def calculate_fitness(self):
        return
    
    def select(self):
        return
    
    def crossover_and_mutations(self):
        return
    
    def crossover(self):
        return
    
    def mutation(self):
        return
    
    def visualization(self):
        return