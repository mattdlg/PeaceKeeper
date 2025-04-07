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
    Deléglise Matthieu & Durand Julie
-------------------------------
Version : 
    4.5 (07/04/2025)
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
        
        self.size_crossover = int(self.dimension[0]/10) # useful only if the array are flattened
        self.square_size = (1, 1) # Arbitrary size of a square to be exchange if crossover_method is "square"

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
        #### Crossovers Parameters ####
        # WARNING : nb_to_retrieve must be even
        nb_iteration = nb_to_retrieve//2 # number of crossover event to do between parents to have the good number of children

        arr_crossing_points = np.zeros(nb_iteration)
        if len(self.dimension) == 1 : # the arrays have been flattened 
            if self.crossover_method == "single-coordinate":
                arr_crossing_points = np.linspace(500, 502, nb_iteration).astype(np.int64)
                # arr_crossing_points = np.random.randint(0, self.dimension[0], nb_iteration).astype(np.int64)

            elif self.crossover_method == "single-point" :
                arr_crossing_points = np.linspace(0.2, 0.8, nb_iteration)*self.dimension
                arr_crossing_points = np.rint(arr_crossing_points).astype(np.int64) # single-points
        
            elif self.crossover_method == "two-points" :
                arr_lower_points = np.linspace(0.25, 0.75, nb_iteration)*self.dimension
                arr_lower_points = np.rint(arr_lower_points).astype(np.int64)
                arr_upper_points = arr_lower_points + self.size_crossover
                arr_crossing_points = np.array([(arr_lower_points[i], arr_upper_points[i]) for i in range(len(arr_lower_points))])
        
        else : # array have more than one dimension
            # For "single" method, choose nb_iteration canal (resp line, column) to exchange between the parents to get nb_iteration*2 children
            if self.crossover_method == "single-canal" : 
                arr_crossing_points = np.random.choice(self.dimension[0], nb_iteration, replace=False) # exchange only some particular canals
            elif self.crossover_method == "single-line" : # ne pas échanger les lignes tout au dessus (pareil dans square ?)
                arr_crossing_points = np.random.choice(self.dimension[1], nb_iteration, replace=False) # exchange only some particular lines
            elif self.crossover_method == "single-column" :
                arr_crossing_points = np.random.choice(self.dimension[2], nb_iteration, replace=False) # exchange only some particular columns
            
            # For "square" method, randomly choose the coordinate of the top left vertice of the square. The size of the square is arbitrarily fixed by self.square_size
            elif self.crossover_method == "square" : # exchange only some particular squares of coordinates
                arr_line_index = np.random.choice(self.dimension[1]-self.square_size[0], nb_iteration, replace=False)
                arr_column_index = np.random.choice(self.dimension[2]-self.square_size[1], nb_iteration, replace=False)
                arr_crossing_points = np.array([(arr_line_index[i], arr_column_index[i]) for i in range(nb_iteration)])

        # faire test square sur qq canal seulement (modif de deux trois features only)

        # BLX-alpha method with exploration parameter of different size :
        arr_explorations = np.linspace(0.2, 0.8, nb_iteration) 

        # Blending method with different percentage of parents' arrays to mix :
        arr_alpha = np.linspace(0.1, 0.45, nb_iteration) # stop before 0.5 because we create two child by crossover -> 0.4 and 0.6 will do exactly the same

        #### Reproduction ####
        list_child = []
        for k in range(nb_iteration):
            ## Crossover ##
            child1, child2 = self.crossover(self.p1, self.p2, 
                                            arr_crossing_points[k], 
                                            arr_explorations[k],
                                            arr_alpha[k])

            ## Mutations ##
            self.modified_mutation(child1)
            self.modified_mutation(child2)

            list_child.append(child1)
            list_child.append(child2)
        
        arr_child = np.asarray(list_child) # return the children as an Array of arrays (easier to use later)
        return arr_child
    
    def crossover(self, parent1, parent2, crossing_point, w, alpha):
        """
        Implementation of a crossing over between two parents arrays. 
        The function exchange some of the vectors' coordinates to 
        create two children vectors. 

        Possibly use five different methods to do so : 
            #### For flattened arrays only ####
            - single-point : exchange every coordinates after an index (included).
            - two-points : exchange every coordinates between the lower and upper bound (included).

            #### For 3D arrays only ####
            - single-canal : exchange every coordinates in a particular canal of the arrays
            - single-line : exchange every coordinates in a particular line of the arrays
            - single column : exchange every coordinates in a particular column of the arrays
            - square : exchange every coordinates in a particular square (same square in each canal) of the arrays

            #### For both types ####
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
            Array of dimension n representing the first parent.
        parent2 : np.array
            Array of dimension n representing the second parent.
        

        Returns
        -------
        child1 : np.array
            Array of dimension n representing the first child.
        child2 : np.array
            Array of dimension n representing the second child.

        """
        size = len(self.dimension)
        # print(self.crossover_method)
        if size == 1 and self.crossover_method == "single-coordinate" : # Only exchange a single coordinate
            child1, child2 = parent1.copy(), parent2.copy()
            print(crossing_point)
            child1[crossing_point] = 100
            child2[crossing_point] = 100

        elif size == 1 and self.crossover_method == "single-point" : # exchange every coordinates after a random index, including this index.
            child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
            child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

        elif size == 1 and self.crossover_method == "two-points" : # exchange every coordinates between two indexes, including them
            low_crossing_point = crossing_point[0] # lower bound can be any index in the range of the size of the vectors
            high_crossing_point = crossing_point[1] # upper bound cannot be inferior to the lower one
            
            child1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
            child2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))
        

        elif size > 1 and self.crossover_method == "single-canal": # échanger plus de canaux
            child1, child2 = parent1.copy(), parent2.copy()
            # Only exchange one canal :
            child1[crossing_point, :, :] = parent2[crossing_point, :, :]
            child2[crossing_point, :, :] = parent1[crossing_point, :, :]
        
        elif size > 1 and self.crossover_method == "single-line": 
            child1, child2 = parent1.copy(), parent2.copy()
            # only exchange one line
            child1[:, crossing_point, :] = parent2[:, crossing_point, :]
            child2[:, crossing_point, :] = parent1[:, crossing_point, :]

        elif size > 1 and self.crossover_method == "single-column":
            # only exchange one column
            child1, child2 = parent1.copy(), parent2.copy()
            child1[:, :, crossing_point] = parent2[:, :, crossing_point]
            child2[:, :, crossing_point] = parent1[:, :, crossing_point]

        elif size > 1 and self.crossover_method == "square": # plusieurs carrés / par canaux # blur(filtre) sur les contours pour meilleur intégration
            child1, child2 = parent1.copy(), parent2.copy()
            lower_line_index, upper_line_index = crossing_point[0], crossing_point[0]+self.square_size[0]
            lower_column_index, upper_column_index = crossing_point[1], crossing_point[1]+self.square_size[1]
            # only exchange one square :
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
            # w == exploration weight
            lower = np.minimum(parent1, parent2) - w * np.abs(parent1 - parent2) # lower = minimum value between the two parents minus w times the difference between them
            upper = np.maximum(parent1, parent2) + w * np.abs(parent1 - parent2) # upper = maximum value between the two parents plus w times the difference between them
            # Randomly creates coordinate between the lower and upper limit (explore more than the range of the parents' values) :
            child1 = np.random.uniform(lower, upper)
            child2 = np.random.uniform(lower, upper)

        elif self.crossover_method == "blending": # linear combination of the coordinates
            # child1 keep alpha*100% of parent1 and (1-alpha)*100% of parent2, and inversely for child2 :
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
        mask = np.random.random_sample(self.dimension) < self.mutation_rate # only mutate with a certain probability
        mutations = np.random.normal(0, self.sigma_mutation, self.dimension) # mutation are drawn from a normal distribution
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
        if len(self.dimension) > 1 : # in case the array is not flattened, apply the same mutation on each canal
            mask = np.random.random_sample(self.dimension[1:]) < self.mutation_rate # only mutate with a certain probability
            mutations = np.random.normal(0, self.sigma_mutation, self.dimension[1:]) # mutation are drawn from a normal distribution
            for i in range(self.dimension[0]):
                chr[i][mask] += mutations[mask] # same mutation on each canal of the matrix
        
        else : # if its flattened, just use the basic method of mutation
            self.mutation(chr)
        # chr[:][mask] += mutations[mask] # add random component to the vectors when its coordinates have mutated.



def test_mutation(picture1, picture2, p_mut=0.5, s_mut=1):
    """
    Test Function of the mutation method of the GA.

    """
    ga = GeneticAlgorithm(picture1.flatten(order="C"), picture2.flatten(order="C"), nb_to_retrieve=6, crossover_method="single-line", 
                          mutation_rate=p_mut, sigma_mutation=s_mut)
    
    np.random.seed(42)
    chr = np.random.rand(2,2,2) * 10
    chr = chr.flatten(order="C")
    
    apply_mutation(chr,ga)

def apply_mutation(chr, ga):
    """
    >>> np.random.seed(42)
    >>> a = np.random.rand(8) * 10
    [3.74540119 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]
    >>> test_mutation(a)
    [ 3.74540119  9.50714306  5.59502159  5.98658484  1.5601864   1.87419254
    -0.32718795  7.24945776]
    """
    print("Before mutation: ")
    print(chr) 
    ga.modified_mutation(chr)
    print("After mutation: ")
    print(chr)
    print("\n")


def test_crossover(picture1, picture2, point_of_crossover, w, alpha, method):
    """
    Test Function of the crossover method of the GA

    >>> np.random.seed(42)
    >>> a = np.random.rand(8) * 10
    [3.74540119 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]
    >>> b = np.random.rand(8) * 10
    [6.01115012 7.08072578 0.20584494 9.69909852 8.32442641 2.12339111
    1.81824967 1.8340451 ]
    >>> test_crossover(a,b,1, 0.5, 0.3, "single-point")
    [3.74540119 7.08072578 0.20584494 9.69909852 8.32442641 2.12339111
    1.81824967 1.8340451 ]
    [6.01115012 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]
    """
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
    """
    Main Function to run the GA class and retrieve its solutions 

    Parameters
    ----------
    targets : list
        List of the arrays of the parents to give to the GA.
    nb_solutions : int
        Number of children to create using crossover and mutation in the GA
    crossover_method : string
        Method of crossover to use to mix coordinate of parents and get new children
    mutation_rate : float
        Probability of mutation of each coordinate of an array
    sigma_mutation : float
        Standard deviation of the normal distribution used to create random vectors of mutation.

    Returns
    -------
    solutions : np.array
        Array of arrays of the solution of the GA 

    """
    # dimensions = targets[0].shape
    # flat_targets = [t.flatten(order="C") for t in targets] # essayer sans aplatir

    ga = GeneticAlgorithm(targets[0], targets[1], nb_to_retrieve=nb_solutions, crossover_method=crossover_method, 
                          mutation_rate=mutation_rate, sigma_mutation=sigma_mutation)
    
    solutions = ga.solutions
    # solutions = solutions.reshape((-1, *dimensions), order="C") 
    return solutions

if __name__ == "__main__" :
    print(__name__)
    
    np.random.seed(42)
    target = np.random.rand(2,2,2) * 10
    target2 = np.random.rand(2,2,2) * 10
    
    # test_mutation(target, target2)
    test_crossover(target.flatten(order="C"), target2.flatten(order="C"), 1, 0.5, 0.3, "single-point")

    """solutions = run_ga([target, target2], nb_solutions=6, crossover_method="blending", mutation_rate=0.5, sigma_mutation=1)


    print("---- Parents ----")
    print(target)
    print(target2)
    print("---- Children ----")
    for s in solutions :
        print(s)"""




    
