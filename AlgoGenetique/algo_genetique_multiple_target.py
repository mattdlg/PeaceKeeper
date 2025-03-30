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
    3.3 (30/03/2025)
"""

#### Libraries ####
import numpy as np
import matplotlib.pyplot as plt
from random import choices

from numba import njit
from joblib import Parallel, delayed
from scipy.optimize import minimize

import time

#### helper function ####
@njit(fastmath=True)
def fast_norm(x):
    """
    This fonction compute the euclidian norm of the vector given as parameter.
    The computation is made faster by using the numba library.
    It aims at decreasing time of computation of Euclidian distances 
    in the Genetic Algorithm (GA).

    Parameters
    ----------
    x : np.array
        Numpy array representing a matrix. In the case of the GA,
        it will contain the vectors between current population positions
        and targets' positions. It is of the shape (size_pop, nb_targets, nb_of_coordinates)

    Returns
    -------
    np.array 
        Array containing the values of the Euclidian norm (norm 2) of these vectors.
        It is all the euclidian distances between each combination of individual and target

    """
    return -np.sqrt(np.sum(x ** 2, axis=2))

class GeneticAlgorithm():
    """
    Class GeneticAlgorithm

    This algorithm aims at creating a set of images similar to one or more target(s) picture(s) using 
    a genetic algorithm. 
    These images are all representing as flattened vectors of dimension n, which are the 
    representation of the images in the latent space of an autoencoder.

    """
    def __init__(self, target_list, max_iteration, size_pop, nb_to_retrieve, stop_threshold, selection_method,
                 crossover_proba, crossover_method, mutation_rate, sigma_mutation, mutation_method):
        """
        Creation of an instance of the GeneticAlgorithm class.

        Parameters
        ----------
        target_list : list
            List of vectors of size n representing the targets photos in the latent space of the autoencoder. 
        max_iteration : int
            Maximal number of generation to obtain the final population
        size_pop : int
            Number of individuals in the population at each generation (constant size)
        nb_to_retrieve : int
            Number of solution of the Genetic Algorithm we want to obtain at the end.
        stop_threshold : float
            Arbitrary minimal condition (distance) to consider a solution to be close enough to one of the targets.
        selection_method : string
            Method used to select best/random individuals at each generation. 
            Default is threshold. Can also be Fortune_Wheel or tournament.
        crossover_proba : float ([0;1])
            Probability of a crossover happening between two parents at each generation
        crossover_method : string
            Method used to cross coordinates of two parents.
            Default is single-point. Can also be two-points, uniform, BLX-alpha or max_diversity.
        mutation_rate : float ([0;1]) or couple of floats
            Probability of mutation of each coordinate (gene) of a vector (chromosome)
            If mutation_method is "constant", must be a single float. Else, must be a couple of floats,
            one for badly fitted individuals and one for nicely fitted individuals.
        sigma_mutation : float
            Standart deviation of the normal distribution defining the random component added during a mutation.
        mutation_method : string 
            Method used to mutate coordinates of an individual.
            Default is constant. Can also be adaptive.

        Returns
        -------
        None

        """
        self.target_photos = np.asarray(target_list) # convert list of targets into a single array (easier to use for computation of euclidian distances)
        self.dimension = len(self.target_photos[0]) # dimension of the vector space = "number of gene of one individual"

        self.max_iteration = max_iteration # max number of generations

        self.population = self.create_random_init_pop(size_pop) # array : initial population from which the evolutionary process begins
        self.generation = None # list : selected population based on fitness at each generation
         
        self.m = nb_to_retrieve # nb of solutions we want to have at the end of the GA
        self.threshold = stop_threshold
        self.count_generation = 0 # to count the number of generations and plot it afterwards. 

        self.dico_fitness = {} # dictionnary to memorize the fitness values of the population at each generation
        # this dictionnary will have the form {n° generation : [fitness_indiv1, fitness_indiv2, ...], ...}
        self.solution = [] # list of the best approximation of the targets at the end the GA.

        # Parameters for selection, crossover and mutation
        self.selection_method = selection_method
        self.crossover_proba = crossover_proba
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.sigma_mutation = sigma_mutation
        self.mutation_method = mutation_method

    def create_random_init_pop(self, size_pop):
        """
        Creation of a list of vectors to start the evolutionnary process.
        To do so, randomly create vectors from a uniform distribution having for limits
        the highest coordinate value in targets. 

        Parameters
        ----------
        size_pop : int 
            Number of individuals in the initial population
        
        Also use the dimension of the vector space (self.dimension)

        Returns
        -------
        init_population : np.array
            Array of the vectors of the initial individuals generated randomly

        """
        limit = np.max(self.target_photos)
        init_population = np.random.uniform(-limit, limit, (size_pop, self.dimension))
        # init_population = np.random.normal(0, limit, (size_pop,self.dimension)) 

        return init_population
    
    def calculate_fitness_without_numba(self):

        """
        Compute and store fitness of each individual
        Here, fitness is defined as the smallest value of distances of the individual 
        with each target.

        Parameters
        ----------
        None 
            We use the euclidian distance to compute fitness 

        Returns
        -------
        list
            List of the fitness of each individual in the same ordre as given by self.population.

        """
        # We use self.population[:, np.newaxis, :] instead of just self.population to ensure
        # that we can compute a euclidian distance for each couple individual-target.
        # For the same purpose, we use axis=2, as the shape of the vector
        # self.population[:, np.newaxis, :] - self.target_photos is (size_pop, nb_targets, nb_of_coordinates)
        # so we have to sum on the last axis to get one value for each couple individual-target.

        distances = -np.linalg.norm(self.population[:, np.newaxis, :] - self.target_photos, axis=2) 

        # as fitness is defined as a negative value 
        # (so that targets have fitness 0 and the fitness of the pop increase until it reach O)
        # the smallest distance is actually the maximal negative value, so we use np.max.
        # Then, we use axis=1 to have one value per individual (shape of distances is (size_pop, nb_targets)).

        return np.max(distances, axis=1).tolist() 
    
    def calculate_fitness_with_numba(self):
        """
        Same function as calculate_fitness_without_numba but use 
        the helper function fast_norm accelerated with numba.

        """
        distances = fast_norm(self.population[:, np.newaxis, :] - self.target_photos)
        return np.max(distances, axis=1).tolist()
    
    def parallel_calculate_fitness(self):
        """
        Same function as calculate_fitness_without_numba but use 
        Parallelisation to accelerate the process.

        """
        fitness = Parallel(n_jobs=-1)(delayed(np.linalg.norm)(indiv - self.target_photos, axis=1) for indiv in self.population)
        return np.max(-np.array(fitness), axis=1).tolist()
    
    def calculate_individual_fitness(self, indiv):
        """
        Function to calculate the fitness of a single individual given as parameter.
        As in calculate_fitness_without_numba, fitness is defined as the smallest distance 
        (max negative value) between this individual and each of the target.

        Parameters 
        ----------
        indiv : np.array
            Vector of a given individual of the population
        
        Returns 
        -------
        float
            Euclidian distance between the parameter and the target vector

        """ 
        fitness_values = -np.linalg.norm(indiv - self.target_photos, axis=1)
        return np.max(fitness_values)
    
    def select(self, nb_generation, criteria = "threshold"):

        """
        Select an even number of individuals according to their fitness. This set of individuals will then 
        endure crossover and mutation.
        
        Possibly use three different criteria to do so : 
            - threshold : choose only the individuals that have a fitness greater than the average one.
            - Fortune_Wheel : Choose individuals randomly, with probabilities proportional to their fitness.
            - tournament : Choose best individuals in smaller random parts of the population.

        Parameters
        ----------
        nb_generation : int
            The index of the generation the algorithm is currently in.
        
        criteria : string
            Name of the criteria used for the selection. Default is "threshold".
            Other criteria is "Fortune_Wheel" or "tournament".

        Returns
        -------
        generation : list
            List of individuals selected to pursue algorithm.

        """
        fitness_values = self.dico_fitness[nb_generation] # fitness of individuals of the current generation

        if criteria == "threshold" :
            generation = []
            max_fitness_after_threshold = 0
            pop = None
            
            fitness_avg = np.mean(fitness_values) # Compute average fitness
            for i in range(len(self.population)) : 
                if  fitness_values[i] >= fitness_avg : # Selection of the individuals
                    generation.append(self.population[i])
                else : 
                    if fitness_values[i] > max_fitness_after_threshold : # Store the individual that is closer to the average fitness in case the number of individuals selected isn't pair.
                        max_fitness_after_threshold, pop = fitness_values[i], self.population[i]

            if len(generation)%2 != 0 : # if we don't have a pair number of individuals
                generation.append(pop)

            return generation

        elif criteria == "Fortune_Wheel" :

            proba_individuals = []
            
            #### Decomment next lines if you want to ensure you keep the 10% best individuals
            """
            elite_size = int(len(self.population) * 0.1)  # keep the 10 % best
            sorted_indices = np.argsort(fitness_values)[::-1]
            elite = [self.population[i] for i in sorted_indices[:elite_size]]
            fitness_array = np.array(fitness_values)
            fitness_array = fitness_array[sorted_indices[elite_size:]] # fitness values of non elite population
            """

            min_fitness = np.min(fitness_values) # compute the most negative fitness -> furthest away from target
            transformed_fitness = np.abs(np.array(fitness_values) - min_fitness) # translate value so that the closest fitness to 0 has the highest new positive value

            # Compute the probabilities of choosing each individual. 
            # The highest their transformed fitness, the highest their probabilities of being chosen.
            # We use the square root (**0.5) to crush the differences that are too marked. 
            # This allows worse individuals to have better chances to survive, to avoid stagnation of the GA.
            # ensure we do not divide by 0 : when all transformed_fitness are 0, probabilities are uniform : 
            # same chances to choose any individual
            proba_individuals = transformed_fitness ** 0.5 / np.sum(transformed_fitness ** 0.5) if np.sum(transformed_fitness) != 0 else np.ones_like(transformed_fitness) / len(transformed_fitness)

            if len(self.population)%2 == 0 : # Choose an even number of individual in the population
                nb_tirages = len(self.population)//2
            else : 
                nb_tirages = len(self.population)//2 + 1

            # If we already kept the best individuals, we need to keep a lesser amount in the rest of the pop :
            # nb_tirages -= elite_size 

            index_choice = np.random.choice(len(self.population), size = nb_tirages, replace = False, p = proba_individuals) # list of the selected individuals according to their probability
            # index_choice = np.random.choice(sorted_indices[elite_size:], size = nb_tirages, replace = False, p = proba_individuals)
            generation = [self.population[k] for k in index_choice]
            # generation = generation + elite

            # to replace one of the value with the best indiv
            """
            max_indice = np.argsort(fitness_values)[-1]
            elite = self.population[max_indice] 
            generation[-1] = elite
            """

            return generation

        elif criteria == "tournament":
            fitness_array = np.array(fitness_values)
            tournament_size = 4 # number of individuals that will compete between each other to be selected
            
            if len(self.population)%2 == 0 : # Choose an even number of individual in the population
                nb_tirages = len(self.population)//2
            else : 
                nb_tirages = len(self.population)//2 + 1

            selected = []
            for _ in range(nb_tirages): 
                indices = np.random.choice(len(self.population), tournament_size, replace=False) # choose randomly some individuals
                best_idx = indices[np.argmax(fitness_array[indices])]
                selected.append(self.population[best_idx]) # select best individual in these, depending on their fitness

            return selected
        else : 
            print("Error, unknown method")
    
    def crossover_and_mutations(self):
        """
        Last step of each loop of the genetic algorithm : 
        - breeding between parents (crossover) to give a child population, 
        - mutation on these children

        Parameters
        ----------
        None 

        Make use of : 
        self.generation : list
            List of the vectors of the selected individuals in the population
        self.crossover_proba : float
            Probability of crossing over between two parents (between 0 and 1)

        Returns
        -------
        new_population : np.array
            Array of the vectors of the new population composed of the parents 
            (best individuals from the previous generation) and their children.

        """
        new_population = self.generation[:] # new population will also contained the selected parents from the previous generation
        for i in range(0, len(self.generation), 2): # Caution : this imply that the generation contains an even number of parents 
            parent1 = self.generation[i]
            parent2 = self.generation[i+1]
            
            # Crossovers
            if np.random.random_sample() < self.crossover_proba :
                child1, child2 = self.crossover(parent1, parent2, method=self.crossover_method)
            else : # no crossing over : children are equal to parents (before mutation)
                child1, child2 = np.copy(parent1), np.copy(parent2)
            
            # Mutations
            self.mutation(child1, self.sigma_mutation, method=self.mutation_method)
            self.mutation(child2, self.sigma_mutation, method=self.mutation_method)
            new_population.append(child1)
            new_population.append(child2)
            
        new_population = np.array(new_population)
        return new_population
    
    def crossover(self, parent1, parent2, method = "single-point"):
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
            - max_diversity : blend coordinates of the two parents (linear combination) using a factor 
            alpha defining the percentage of each parents is kept (ex : child1 = 0.7parent1 + 0.3parent2).
            Alpha is set as a random value (in [0.2, 0.8]) to ensure more diversity : 
            not the same blending at each generation.

        Parameters
        ----------
        parent1 : np.array
            Array of dimension n representing the first parent (vector).
        parent2 : np.array
            Array of dimension n representing the second parent (vector).
        method : string
            Name of the method used for the crossing over. Default is "single-point".
            Other methods are "two-points", "uniform", "BLX-alpha" and "max_diversity".

        Returns
        -------
        child1 : np.array
            Array of dimension n representing the first child (vector).
        child2 : np.array
            Array of dimension n representing the second child (vector).

        """
        if method == "single-point" : # exchange every coordinates after a random index, including this index.
            crossing_point = np.random.randint(1, self.dimension) # self.dimension = size of the vectors = dimension of the vector space
            child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
            child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

        elif method == "two-points" : # exchange every coordinates between two indexes, including them
            low_crossing_point = np.random.randint(0, self.dimension) # lower bound can be any index in the range of the size of the vectors
            upper_bound = min(low_crossing_point + (self.dimension-1), self.dimension) # to be sure that not every coordinates are exchanged (meaning that children = parents), we ensure that at least one stay the same
            high_crossing_point = np.random.randint(low_crossing_point, upper_bound) # upper bound cannot be inferior to the lower one
            child1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
            child2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))

        elif method == "uniform" : # randomly choose from which parents a coordinate will be 
            child1, child2 = np.array([0.0 for _ in range(self.dimension)]), np.array([0.0 for _ in range(self.dimension)])
            mask = np.random.randint(0, 2, size=self.dimension, dtype=bool) # toss a coin for each coordinate : heads = coord from p1, tails = coord from p2
            child1, child2 = np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)

            if np.array_equal(child1, parent1) : # if no exchange were made, while we want at least one when there is a crossover:
                random_index = np.random.randint(self.dimension) # randomly choose a coordinate to exchange to ensure that children differ from parents
                child1[random_index] = parent2[random_index]
                child2[random_index] = parent1[random_index]
        
        elif method == "BLX-alpha": # Blend Alpha crossover method
            alpha = 0.5 # degree of exploration (increase of the range of values)

            # compute lower and higher bound for each coordinate 
            lower = np.minimum(parent1, parent2) - alpha * np.abs(parent1 - parent2) # lower = minimum value between the two parents minus alpha times the difference between them
            upper = np.maximum(parent1, parent2) + alpha * np.abs(parent1 - parent2) # upper = maximum value between the two parents plus alpha times the difference between them
            child1 = np.random.uniform(lower, upper)
            child2 = np.random.uniform(lower, upper)

        elif method == "max_diversity": 
            alpha = np.random.uniform(0.2, 0.8)  # Randomly varies the intensity of the blending
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

            # add a random component to explore outside the range of value of the parents = increase diversity.
            # Already done with the mutations but can be useful to ensure more diversity. 
            return child1, child2
            # return child1 + np.random.normal(0, 0.2, size=parent1.shape), child2 + np.random.normal(0, 0.2, size=parent2.shape)

        else : 
            print("Error, unknown method")
            child1, child2 = parent1, parent2 # keep children equal to parents if a wrong method is called

        return child1, child2
    
    def mutation(self, chr, sigma_mutation, method = "constant"):
        """
        Implementation of a mutational event on a chromosome/individual (vector).
        Each gene (coordinate) of the chromosome as a probability of mutating depending
        on the mutation rate given in parameter. 

        Mutations are drawn from a normal distribution distributed around 0. 
        Modifying the standard deviation sigma allows to have bigger or smaller mutations.

        Two different methods are implemented : 
            - constant : In this case, there is one single mutation rate given as parameter. 
            Each chromosome in the population have the same probability of mutation for their genes.
            - adaptive : In this case, there is two mutation rates given as parameters : the first
            one is for badly fitted chromosomes, and the second one for well fitted chromosomes. 
            Indeed, when a chromosome is well fitted, its coordinates are already close from the target, 
            meaning that mutating a lot of genes increases the chances of moving away from it. On the contrary,
            a badly fitted chromosome has all interest in having a high mutation probability to change a lot 
            of its coordinates and move closer to the target. Therefore, it can be useful to have two different 
            mutation rates, one smaller for the good fits and one higher for the bad fits, to converge more
            rapidly toward the target.

        Parameters
        ----------
        chr : 
        mutation_rate : float or tuple
            probability of mutation of a gene (between 0 and 1).
            If method is "constant", mutation rate is a unique float.
            If method is "adaptive", mutation rate is a tuple of two floats : (rate_for_low_fitness, rate_for_high_fitness)
        sigma_mutation : float
            Standard deviation of the normal distribution from which are drawn mutations.
        method : str
            Method use for the mutation rate. Default is "constant". Other method is "adaptive".

        Returns
        -------
        None
            chr is modified directly by the function and thus do not need to be returned.

        """
        if method == "constant": # in the constant case, there is a unique mutation rate that is the same for every chromosomes
            proba_mutation = self.mutation_rate
            
        elif method == "adaptive": # in the adaptive case, the mutation rate depends on the fitness of the chromosome
            fit_avg = np.mean(self.dico_fitness[self.count_generation]) # average fitness among the curent generation
            fit_chr = self.calculate_individual_fitness(chr) # fitness of the current individual
            if fit_chr >= fit_avg : # individual is well fitted compared to the rest of the population 
                proba_mutation = self.mutation_rate[1]
            else : # individual is badly fitted compared to the rest of the population
                proba_mutation = self.mutation_rate[0]

        else : 
            print("Error, unknow method") # if the method is incorrect, do not apply any mutation 
            return 
        
        # add a temperature : the more we advance, the more the individuals are closer to the targets :
        # we therefore decrease the size of mutation at each generation using this factor
        T = max(0.01, 1 - self.count_generation / self.max_iteration) 
        """
        if np.std(self.population, axis=0).mean() < 0.1:  # diversity threshold
            self.sigma_mutation *= 1.2  # increase back mutation size to increase diversity
        """
        mask = np.random.rand(self.dimension) < proba_mutation # only mutate with a certain probability
        mutations = np.random.normal(0, sigma_mutation*T, self.dimension)
        chr[mask] += mutations[mask] # add random component to the vectors when its coordinates have mutated.

    def visualization(self):
        """
        Plot the evolution of the fitness of the population at each generation.

        Parameters
        ----------
        None

        Returns 
        -------
        None 
            Open a new window with the graph.
        """
        index_generations = list(range(1, list(self.dico_fitness.keys())[-1]+1)) # x-axis

        # fitness range :
        best_fitness_values = [np.max(list_fitness) for list_fitness in self.dico_fitness.values()]
        worst_fitness_values = [np.min(list_fitness) for list_fitness in self.dico_fitness.values()]

        plt.figure()
        plt.plot(index_generations, best_fitness_values, label='Best Fitness', color='black') # maximal value of fitness at each generation
        plt.fill_between(index_generations, worst_fitness_values, best_fitness_values, color='gray', alpha=0.5, label='Fitness Range') # Interval of value at each generation
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()
        plt.show()

    def stop_condition(self):
        """
        Stopping condition of the GA : 
        If the m individuals of the population with the highest fitness values 
        all have a fitness superior to an arbitrary threshold (close to 0), stop the algorithms :
        we consider that we are close enough from the targets.

        Parameters 
        ----------
        None 

        Make use of the following parameters of the class
        self.threshold : float
            Max distance until when we consider that we are close from the target

        Returns
        -------
        stop : bool
            True or false, depending on if we need to stop the GA or not.
            True if m individuals respect the stop conditions

        """
        self.solution, max_fitness = self.retrieve_max_fitness_population() #  to retrieve only the m closest individuals to the target

        stop = True
        for val in max_fitness :
            if val < self.threshold : # if at least one of the value do not respect the stop condition, we need to pursue the GA
                stop = False
                break
    
        return stop
    
    def retrieve_max_fitness_population(self):
        """
        Retrieve the m more fitted solutions in the population, 
        which are the m closest vectors to the targets.

        Parameters
        ----------
        None

        Make use of the following parameters from the class :
        self.m : int 
            Number of vectors to retrieve
        self.dico_fitness : dict
            Values of fitness of each individual at each generation
        self.population : np.array
            Array of the vectors of each individual of the population

        Returns
        -------
        population_max : list
            List of the m vectors having the best fitnesses
        fitness_max : np.array
            Array of the m highest fitness values

        """
        last_fitnesses = list(self.dico_fitness.values())[-1][:]

        indices_max = np.argsort(last_fitnesses)[-self.m:][::-1] # finding the indices corresponding to the m highest values, in decreasing order
        fitness_max = np.sort(last_fitnesses)[-self.m:][::-1] # m highest values of fitness (used in the stop condition)

        population_max = [self.population[i] for i in indices_max]
        return population_max, fitness_max
    
    def main_loop(self):
        """
        Main Loop of the Genetic Algorithm.
        The loop represent the evolution of the population over generations.
        It follows the pattern : compute fitness --- selection --- crossover --- mutations

        Parameters
        ----------
        None

        Returns
        -------
        self.solution : list
            List of the vectors solution of the Genetic Algorithm
        """
        
        while self.count_generation < self.max_iteration :
            self.count_generation += 1 # next generation
            self.dico_fitness[self.count_generation] = self.calculate_fitness_without_numba()
            if self.stop_condition(): # if we have solutions close enough to the target
                break
            self.generation = self.select(self.count_generation, criteria=self.selection_method) 
            new_population = self.crossover_and_mutations()
            self.population = new_population
            
            # To avoid local minima (NOT WORKING FOR NOW)
            """
            if self.count_generation % 50 == 0:
                self.refine_best_individual()
            """

        # print("Écart-type des coordonnées finales :", np.std(self.solution, axis=0).mean())
        return self.solution
    
    def refine_best_individual(self):
        """
        Gradient descent on best individual (local optimisation)
        to increase convergence and avoir local minima.

        Parameters
        ----------
        None

        Returns
        -------
        None 
            Replace best individuals in the population by the new optimized vectors
        """
        best_idx = np.argmax(self.dico_fitness[self.count_generation])
        best_individual = self.population[best_idx]

        result = minimize(lambda x: np.linalg.norm(x - self.target_photo), best_individual, method="BFGS")
        
        if result.success:
            self.population[best_idx] = result.x  # Remplace par la version optimisée


def varying_target(target, nb_solutions):
    dimensions = target.shape

    list_target = []
    for _ in range(nb_solutions):
        mask = np.random.rand(dimensions[0], dimensions[1], dimensions[2]) < 0.5
        mutations = np.random.normal(0, 1, dimensions)
        new_target = np.copy(target)
        new_target[mask] += mutations[mask]

        list_target.append(new_target)

    print("Écart-type des coordonnées des targets :", np.std(list_target, axis=0).mean())
    return list_target
        
def test_fitness(targets):

    targets = [t.flatten(order = "C") for t in targets]
    ga = GeneticAlgorithm(targets, max_iteration=2000, size_pop=100, nb_to_retrieve=len(targets), stop_threshold=-1, 
                            selection_method="Fortune_Wheel", crossover_proba=0.9, crossover_method="max_diversity", 
                            mutation_rate=(0.5, 0.05), sigma_mutation=0.8, mutation_method="adaptive")

    start = time.time()
    fitness1 = ga.calculate_fitness_without_numba()  # Version NumPy seule
    end = time.time()
    print(f"Temps sans numba: {end - start:.4f} sec")

    start = time.time()
    fitness2 = ga.calculate_fitness_with_numba()  # Version avec numba
    end = time.time()
    print(f"Temps avec numba: {end - start:.4f} sec")

    start = time.time()
    fitness3 = ga.parallel_calculate_fitness()  # Version parallélisée
    end = time.time()
    print(f"Temps avec parallélisation: {end - start:.4f} sec")

def test_BLX_alpha():
    targets = [np.random.rand(4,4) * 10 for _ in range(2)]
    targets = [t.flatten(order = "C") for t in targets]
    ga = GeneticAlgorithm(targets, max_iteration=2000, size_pop=100, nb_to_retrieve=len(targets), stop_threshold=-1, 
                            selection_method="Fortune_Wheel", crossover_proba=0.9, crossover_method="max_diversity", 
                            mutation_rate=(0.5, 0.05), sigma_mutation=0.8, mutation_method="adaptive")
    
    parent1 = targets[0]
    parent2 = targets[1]
    child1, child2 = ga.crossover(parent1, parent2, method="BLX-alpha")
    print(parent1)
    print(parent2)
    print("-------------------")
    print(child1)
    print(child2)

def ga_with_multiple_targets(targets):

    targets = [t.flatten(order = "C") for t in targets]
    ga = GeneticAlgorithm(targets, max_iteration=2000, size_pop=100, nb_to_retrieve=len(targets), stop_threshold=-1, 
                            selection_method="Fortune_Wheel", crossover_proba=0.9, crossover_method="max_diversity", 
                            mutation_rate=(0.5, 0.05), sigma_mutation=0.8, mutation_method="adaptive")
    
    solutions = ga.main_loop()
    ga.visualization()
    
    print("Écart-type des coordonnées finales :", np.std(solutions, axis=0).mean())
    for s in solutions: 
        print(f" norm : {np.max(-np.linalg.norm(s-np.asarray(targets), axis=1))}")

    return solutions

def run_ga(i, targets, nb_solutions):
    partial_targets = [t[i, :, :].flatten(order="C") for t in targets]
    ga = GeneticAlgorithm(partial_targets, max_iteration=2000, size_pop=60, nb_to_retrieve=nb_solutions, stop_threshold=-1, 
                            selection_method="Fortune_Wheel", crossover_proba=0.9, crossover_method="max_diversity", 
                            mutation_rate=(0.5, 0.05), sigma_mutation=0.8, mutation_method="adaptive")
    print(i)
    return ga.main_loop()

def ga_multiple_targets_separated(targets):
    dimensions = targets[0].shape
    print(dimensions)
    solutions = Parallel(n_jobs=-1)(delayed(run_ga)(i, targets, len(targets)) for i in range(dimensions[0]))

    reconstructed_solutions = np.stack(solutions, axis=1).reshape((-1, *dimensions), order="C")

    print("Écart-type des coordonnées finales :", np.std(solutions, axis=0).mean())
    for s in reconstructed_solutions: 
         print(f" norm : {np.max(-np.linalg.norm(s-np.asarray(targets), axis=1))}")
    
    return reconstructed_solutions


def create_multiple_target_from_pictures(photos, nb_solutions):
    alphas = np.linspace(0.1, 0.9, nb_solutions)
    targets = []
    for a in alphas :
        new_target = a * photos[0] + (1 - a) * photos[1]
        targets.append(new_target)

    return targets

def run_multiple_ga(targets):
    dimensions = targets[0].shape
    list_solutions = []
    for picture in targets :
        solution = Parallel(n_jobs=-1)(delayed(run_ga)(i, [picture], 1) for i in range(dimensions[0]))
        reconstructed_solution = np.stack(solution, axis=1).reshape((-1, *dimensions), order="C")

        print(-np.linalg.norm(reconstructed_solution - picture))

        list_solutions.append(reconstructed_solution)
        
    array_solutions = np.asarray(list_solutions)
    return array_solutions

def normalization(v):
    """
    get a vector in [0,1] space
    """
    norm_vector = (v - np.min(v)) / (np.max(v)-np.min(v))
    return norm_vector

if __name__ == "__main__" :
    print(__name__)
    target = np.random.rand(128,8,8) * 20
    norm_target = normalization(target)
    # list_targets = varying_target(target, 6)
    target2 = np.random.rand(128,8,8) * 20
    norm_target2 = normalization(target2)
    # photos = [target, target2]
    photos = [norm_target, norm_target2]
    list_targets = create_multiple_target_from_pictures(photos, 6)

    # test_fitness(list_targets)
    # ga_with_multiple_targets(list_targets)

    # ga_multiple_targets_separated(list_targets)
    # run_multiple_ga(list_targets)
    test_BLX_alpha()



    
