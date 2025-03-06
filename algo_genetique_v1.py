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
import numpy as np

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
        self.dimension = len(target)

    def create_random_init_pop(self):
        return
    
    def calculate_fitness(self):
        return
    
    def select(self):
        return
    
    def crossover_and_mutations(self):
        return
    
    def crossover(self, parent1, parent2, method = "single-point"):
        """
        Implementation of a crossing over between two parents vectors. 
        The function exchange some of the vectors' coordinates to 
        create two children vectors. 

        Possibly use three different methods to do so : 
            - single-point : exchange every coordinates after this index (included).
            - two-points : exchange every coordinates between the lower and upper bound (included).
            - uniform : for each coordinate : toss a coin to know if it will be from the first or 
            the second parent (if no coordinate were exchanged, randomly choose one to exchange).

        Parameters
        ----------
        parent1 : np.array
            Array of dimension n representing the first parent (vector).
        parent2 : np.array
            Array of dimension n representing the second parent (vector).
        method : string
            Name of the method used for the crossing over. Default is "single-point".

        Returns
        -------
        children1 : np.array
            Array of dimension n representing the first child (vector).
        children2 : np.array
            Array of dimension n representing the second child (vector).

        """
        if method == "single-point" : # exchange every coordinates after a random index, including this index.
            crossing_point = np.random.randint(1, self.dimension) # self.dimension = size of the vectors = dimension of the vector space
            print(crossing_point)
            children1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
            children2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

        elif method == "two-points" : # exchange every coordinates between two indexes, including them
            low_crossing_point = np.random.randint(0, self.dimension) # lower bound can be any index in the range of the size of the vectors
            upper_bound = min(low_crossing_point + (self.dimension-1), self.dimension) # to be sure that not every coordinates are exchanged (meaning that children = parents), we ensure that at least one stay the same
            high_crossing_point = np.random.randint(low_crossing_point, upper_bound) # upper bound cannot be inferior to the lower one
            print(low_crossing_point, high_crossing_point)
            children1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
            children2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))

        elif method == "uniform" : # randomly choose from which parents a coordinate will be 
            children1, children2 = np.array([0 for _ in range(self.dimension)]), np.array([0 for _ in range(self.dimension)])
            for i in range(self.dimension):
                p = np.random.randint(0,2) # tossing a coin : result = 0 or 1
                if p == 0 : # coordinate of child i come from parent i
                    children1[i] = parent1[i]
                    children2[i] = parent2[i]
                else : # coordinate of child i come from the other parent
                    children1[i] = parent2[i]
                    children2[i] = parent1[i]

            if np.array_equal(children1, parent1) : # if no exchange were made, while we want at least one when there is a crossover:
                random_index = np.random.randint(self.dimension) # randomly choose a coordinate to exchange to ensure that children differ from parents
                children1[random_index] = parent2[random_index]
                children2[random_index] = parent1[random_index]
        else : 
            print("Error, unknown method")
            children1, children2 = parent1, parent2 # keep children equal to parents if a wrong method is called

        return children1, children2
    
    def mutation(self):
        return
    
    def visualization(self):
        return
    
def test_crossing_over(ga, method = "single-point"):
    p1 = np.array([1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,0])
    p2 = np.array([1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,0])
    c1, c2 = ga.crossover(p1,p2, method)
    print("Parents: ")
    print(p1)
    print(p2)
    print("Children: ")
    print(c1)
    print(c2)

if __name__ == "__main__" :
    target = np.array([1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,0])
    ga = GeneticAlgorithm(target)
    test_crossing_over(ga, "uniform")