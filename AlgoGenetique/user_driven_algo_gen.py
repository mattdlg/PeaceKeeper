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
    4.3 (31/03/2025)
"""

#### Libraries ####
import numpy as np

class GeneticAlgorithm():
    """
    Class GeneticAlgorithm

    This algorithm aims at creating a set of images similar to one or more target(s) picture(s) using 
    genetic algorithm technics of crossover and mutation. 
    The rest of the GA steps (initialisation, selection, stop condition)
    are directly done by the user in a graphical interface (see inference.py)
    These images are all represented as small arrays, which are the 
    representation of the images in the latent space of an autoencoder.

    """
    
    def __init__(self, picture1, picture2, nb_to_retrieve, mutation_rate, sigma_mutation, crossover_method = "square"):
        """
        Creation of an instance of the GeneticAlgorithm class.

        Parameters
        ----------
        picture1 : np.array
           Array of size (m,n,p) encoding an image.
        picture2 : np.array
           Array of size (m,n,p) encoding a second image.
        nb_to_retrieve : int
            Number of solutions (images to create) of the Genetic Algorithm we want to obtain at the end.
        crossover_method : string
            Method used to cross coordinates of the two parents images.
            Default is "square". Can also be "single-canal", "single-line", 
            "single-column", "uniform", "BLX-alpha" or "blending".
        mutation_rate : float ([0;1])
            Probability of mutation of each coordinate (gene) of an array (chromosome).
        sigma_mutation : float
            Standard deviation of the normal distribution defining the random component added during a mutation.

        Returns
        -------
        None

        """
        #### Initial images ####
        self.p1 = picture1
        self.p2 = picture2

        self.dimension = self.p1.shape # dimension of the vector space = "number of gene of one individual"

        #### Crossover and mutation parameters ####
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.sigma_mutation = sigma_mutation
        
        self.square_size = (2, 2) # Arbitrary size of a square to be exchange if crossover_method is "square"

        self.solutions = self.crossover_and_mutations(nb_to_retrieve) # solutions of the GA.

    def crossover_and_mutations(self, nb_to_retrieve):
        """
        Last step of each loop of a Genetic algorithm :
        - breeding between parents (crossover) to give a child population, 
        - mutation on these children

        Parameters
        ----------
        nb_to_retrieve : int
            Number of solutions to create by crossing and mutating initial images. 

        Make use of : 
        self.p1, self.p2, self.dimension
        self.crossover_method, self.square_size

        Returns
        -------
        arr_child : np.array
            Array of arrays of the new population composed of the children images 
            created by crossing coordinates of parents and then adding random mutations.

        """
        # Crossovers
        nb_iteration = nb_to_retrieve//2 # nb_to_retrieve must be even

        """
        #### Decomment this if the array are flattened ####
        arr_crossing_points = np.linspace(0.2, 0.8, nb_iteration)*self.dimension
        arr_crossing_points = np.rint(arr_crossing_points).astype(np.int64)
    
        if self.crossover_method == "two-points" :
            arr_lower_points = np.linspace(0.25, 0.75, nb_iteration)*self.dimension
            arr_lower_points = np.rint(arr_lower_points).astype(np.int64)
            arr_upper_points = arr_lower_points + size_crossover
            arr_crossing_points = np.array([(arr_lower_points[i], arr_upper_points[i]) for i in range(len(arr_lower_points))])
        """

        arr_crossing_points = np.zeros(nb_iteration)
        if self.crossover_method == "single-canal" :
            arr_crossing_points = np.random.choice(self.dimension[0], nb_iteration, replace=False) # exchange only some particular canals
        elif self.crossover_method == "single-line" : # ne pas échanger les lignes tout au dessus (pareil dans square ?)
            arr_crossing_points = np.random.choice(self.dimension[1], nb_iteration, replace=False) # exchange only some particular lines
        elif self.crossover_method == "single-column" :
            arr_crossing_points = np.random.choice(self.dimension[2], nb_iteration, replace=False) # exchange only some particular columns
        
        elif self.crossover_method == "square" : # exchange only some particular squares of coordinates
            arr_line_index = np.random.choice(self.dimension[1]-self.square_size[0], nb_iteration, replace=False)
            arr_column_index = np.random.choice(self.dimension[2]-self.square_size[1], nb_iteration, replace=False)
            arr_crossing_points = np.array([(arr_line_index[i], arr_column_index[i]) for i in range(nb_iteration)])

        # faire test square sur qq canal seulement (modif de deux trois features only)

        arr_explorations = np.linspace(0.2, 0.8, nb_iteration)

        arr_alpha = np.linspace(0.1, 0.45, nb_iteration) # stop before 0.5 because we create two child by crossover -> 0.4 and 0.6 will do exactly the same

        list_child = []
        for k in range(nb_iteration):
            # Crossover
            child1, child2 = self.crossover(self.p1, self.p2, 
                                            arr_crossing_points[k], 
                                            arr_explorations[k],
                                            arr_alpha[k])

            # Mutations
            self.modified_mutation(child1)
            self.modified_mutation(child2)

            list_child.append(child1)
            list_child.append(child2)
        
        arr_child = np.asarray(list_child)
        return arr_child
    
    def crossover(self, parent1, parent2, crossing_point, w, alpha):
        """
        Implementation of a crossing over between two parents vectors. 
        The function exchange some of the vectors' coordinates to 
        create two children vectors. 

        Possibly use five different methods to do so : 
            - single-point : exchange every coordinates after an index (included).
            - two-points : exchange every coordinates between the lower and upper bound (included).
            - uniform : for each coordinate : toss a coin to know if it will be from the first or 
            the second parent (if no coordinate were exchanged, randomly choose one to exchange).
            - BLX-alpha : Extend the range of values the children can take outside of the parents range, 
            using a factor alpha. It allows more diversity in offsprings to avoid premature convergence 
            of the GA.
            - blending : blend coordinates of the two parents (linear combination) using a factor 
            alpha defining the percentage of each parents is kept (ex : child1 = 0.7parent1 + 0.3parent2).

        Parameters
        ----------
        parent1 : np.array
            Array of dimension n representing the first parent (vector).
        parent2 : np.array
            Array of dimension n representing the second parent (vector).
        method : string
            Name of the method used for the crossing over. Default is "single-point".
            Other methods are "two-points", "uniform", "BLX-alpha" and "blending".

        Returns
        -------
        child1 : np.array
            Array of dimension n representing the first child (vector).
        child2 : np.array
            Array of dimension n representing the second child (vector).

        """

        """
        #### Decomment this if the array are flattened ####
        if self.crossover_method == "single-point" : # exchange every coordinates after a random index, including this index.
            child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
            child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

        elif self.crossover_method == "two-points" : # exchange every coordinates between two indexes, including them
            low_crossing_point = crossing_point[0] # lower bound can be any index in the range of the size of the vectors
            high_crossing_point = crossing_point[1] # upper bound cannot be inferior to the lower one
            
            child1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
            child2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))
        """

        if self.crossover_method == "single-canal": # échanger plus de canaux
            child1, child2 = parent1.copy(), parent2.copy()
            child1[crossing_point, :, :] = parent2[crossing_point, :, :]
            child2[crossing_point, :, :] = parent1[crossing_point, :, :]
        
        elif self.crossover_method == "single-line": 
            child1, child2 = parent1.copy(), parent2.copy()
            child1[:, crossing_point, :] = parent2[:, crossing_point, :]
            child2[:, crossing_point, :] = parent1[:, crossing_point, :]

        elif self.crossover_method == "single-column":
            child1, child2 = parent1.copy(), parent2.copy()
            child1[:, :, crossing_point] = parent2[:, :, crossing_point]
            child2[:, :, crossing_point] = parent1[:, :, crossing_point]

        elif self.crossover_method == "square": # plusieurs carrés / par canaux # blur(filtre) sur les contours pour meilleur intégration
            child1, child2 = parent1.copy(), parent2.copy()
            lower_line_index, upper_line_index = crossing_point[0], crossing_point[0]+self.square_size[0]
            lower_column_index, upper_column_index = crossing_point[1], crossing_point[1]+self.square_size[1]
            child1[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index] = parent2[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index]
            child2[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index] = parent1[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index]

        elif self.crossover_method == "uniform" : # randomly choose from which parents a coordinate will be 
            child1, child2 = np.zeros(self.dimension, dtype=np.float32), np.zeros(self.dimension, dtype=np.float32)
            mask = np.random.randint(0, 2, size=self.dimension, dtype=bool) # toss a coin for each coordinate : heads = coord from p1, tails = coord from p2
            child1, child2 = np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)

            if np.array_equal(child1, parent1) : # if no exchanges were made, while we want at least one when there is a crossover:
                random_index = np.random.randint(self.dimension) # randomly choose a coordinate to exchange to ensure that children differ from parents
                child1[random_index] = parent2[random_index]
                child2[random_index] = parent1[random_index]
        
        elif self.crossover_method == "BLX-alpha": # Blend Alpha crossover method
            # compute lower and higher bound for each coordinate 
            lower = np.minimum(parent1, parent2) - w * np.abs(parent1 - parent2) # lower = minimum value between the two parents minus alpha times the difference between them
            upper = np.maximum(parent1, parent2) + w * np.abs(parent1 - parent2) # upper = maximum value between the two parents plus alpha times the difference between them
            child1 = np.random.uniform(lower, upper)
            child2 = np.random.uniform(lower, upper)

        elif self.crossover_method == "blending": # linear combination of the coordinates
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

        else : 
            print("Error, unknown method")
            child1, child2 = parent1, parent2 # keep children equal to parents if a wrong method is called

        return child1, child2
    

    def mutation(self, chr):
        """
        Implementation of a mutational event on a chromosome/individual (array).
        Each gene (coordinate) of the chromosome as a probability of mutating depending
        on the mutation rate given in parameter of the class. 
        In this version of the mutation method, different mutational vectors are added to
        each canal (first dimension)

        Mutations are drawn from a normal distribution distributed around 0. 
        Modifying the standard deviation sigma allows to have bigger or smaller mutations.

        Parameters
        ----------
        chr : np.array
            Array representing an encoded image
        
        Make use of : 
        self.dimension, self.mutation_rate and self.sigma_mutation

        Returns
        -------
        None
            chr is modified directly by the function and thus do not need to be returned.

        """
        mask = np.random.rand(self.dimension[0], self.dimension[1], self.dimension[2]) < self.mutation_rate # only mutate with a certain probability
        mutations = np.random.normal(0, self.sigma_mutation, self.dimension)
        chr[mask] += mutations[mask] # add random component to the vectors when its coordinates have mutated.

    def modified_mutation(self, chr):
        """
        Implementation of a mutational event on a chromosome/individual (array).
        Each gene (coordinate) of the chromosome as a probability of mutating depending
        on the mutation rate given in parameter of the class. 
        In this version of the mutation method, the same mutational vector is added to 
        each canal of the image (first dimension)

        Mutations are drawn from a normal distribution distributed around 0. 
        Modifying the standard deviation sigma allows to have bigger or smaller mutations.

        Parameters
        ----------
        chr : np.array
            Array representing an encoded image
        
        Make use of : 
        self.dimension, self.mutation_rate and self.sigma_mutation

        Returns
        -------
        None
            chr is modified directly by the function and thus do not need to be returned.

        """
        mask = np.random.rand(self.dimension[1], self.dimension[2]) < self.mutation_rate # only mutate with a certain probability
        mutations = np.random.normal(0, self.sigma_mutation, (self.dimension[1], self.dimension[2]))
        for i in range(self.dimension[0]):
            chr[i][mask] += mutations[mask]
        # chr[:][mask] += mutations[mask] # add random component to the vectors when its coordinates have mutated.



def test_mutation(picture1, picture2):
    ga = GeneticAlgorithm(picture1, picture2, nb_to_retrieve=6, crossover_method="single-line", 
                          mutation_rate=0.5, sigma_mutation=1)
    
    chr = np.random.rand(4,4,3) * 20
    # chr = chr.flatten(order="C")
    print("Before mutation: ")
    print(chr) 
    ga.mutation(chr)
    print("After mutation: ")
    print(chr)
    print("\n")

def test_crossover(picture1, picture2, point_of_crossover, w, alpha, method):
    ga = GeneticAlgorithm(picture1, picture2, nb_to_retrieve=2, crossover_method=method, 
                          mutation_rate=0.5, sigma_mutation=1)
    
    print("---- Parents ----")
    print(picture1)
    print(picture2)
    child1, child2 = ga.crossover(picture1, picture2, point_of_crossover, w, alpha)
    print("---- Children ----")
    print(child1)
    print(child2)

def run_ga(targets, nb_solutions, crossover_method, mutation_rate, sigma_mutation):
    # dimensions = targets[0].shape
    # flat_targets = [t.flatten(order="C") for t in targets] # essayer sans aplatir

    ga = GeneticAlgorithm(targets[0], targets[1], nb_to_retrieve=nb_solutions, crossover_method=crossover_method, 
                          mutation_rate=mutation_rate, sigma_mutation=sigma_mutation)
    
    solutions = ga.solutions
    # solutions = solutions.reshape((-1, *dimensions), order="C") 
    return solutions

if __name__ == "__main__" :
    print(__name__)
    target = np.random.rand(4,4,3) * 20
    target2 = np.random.rand(4,4,3) * 20
    # test_mutation(target, target2)
    ##test_crossover(target, target2, 1, 0.5, 0.3, "single-canal")

    solutions = run_ga([target, target2], nb_solutions=6, crossover_method="blending", mutation_rate=0.5, sigma_mutation=1)


    print("---- Parents ----")
    print(target)
    print(target2)
    print("---- Children ----")
    for s in solutions :
        print(s)




    
