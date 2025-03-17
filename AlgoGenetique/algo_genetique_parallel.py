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
    1.9 (17/03/2025)
"""
import numpy as np
import matplotlib.pyplot as plt
from random import choices

from numba import njit
from joblib import Parallel, delayed
from scipy.optimize import minimize

@njit(fastmath=True)
def fast_norm(x):
    return -np.sqrt(np.sum(x ** 2, axis=1))

class GeneticAlgorithm():
    """
    Class GeneticAlgorithm

    This algorithm aims at creating a set of images similar to a target picture using 
    a genetic algorithm. 
    These images are all representing as vectors of dimension n, which are the 
    representation of the images in the latent space of an autoencoder.

    """
    def __init__(self, target, max_iteration, size_pop, nb_to_retrieve, stop_threshold, selection_method,
                 crossover_proba, crossover_method, mutation_rate, sigma_mutation, mutation_method):
        """
        Creation of an instance of the GeneticAlgorithm class.

        Parameters
        ----------
        target : np.array
            Vector of size n representing the target photo in the latent space of the autoencoder. 
        max_iteration : int
            Maximal number of generation to obtain the final population

        Returns
        -------
        None

        """
        self.target_photo = target
        self.dimension = len(target) # dimension of the vector space = "number of gene of one individual"

        self.max_iteration = max_iteration

        self.population = self.create_random_init_pop(size_pop) # list : initial population from which the evolutionary process begins
        self.generation = None # list : selected population based on fitness at each generation
         
        self.m = nb_to_retrieve # nb of solutions we want to have at the end of the GA
        self.threshold = stop_threshold
        self.count_generation = 0 # to count the number of generations and plot it afterwards. 

        self.dico_fitness = {} # dictionnary to memorize the fitness values of the population at each generation
        self.solution = [] # list of the best approximation of the target at the end the GA.

        self.selection_method = selection_method
        self.crossover_proba = crossover_proba
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.sigma_mutation = sigma_mutation
        self.mutation_method = mutation_method

    def create_random_init_pop(self, size_pop):
        """
        Creation of a list of ten vectors to start the evolution process.

        Parameters
        ----------
        size_pop : int 
            Number of individuals in the initial population
        
        Use the dimension of the given target vector (self.dimension)

        Returns
        -------
        init_population : np.array
            Array of the vectors of the initial individuals generated randomly

        """

        init_population = np.random.uniform(-10, 10, (size_pop, self.dimension))
        # init_population = np.random.normal(0, 2, (size_pop,self.dimension)) 

        """for _ in range(size_pop) : 
            # Attention la vrai population ça sera pas juste des entiers et surtout pas que des 0 et des 1
            init_population.append(np.random.rand(self.dimension)*10) #Generate an individual randomly
        init_population = np.array(init_population)"""

        return init_population
    
    def calculate_fitness(self):

        """
        Compute and store fitness of each individual

        Parameters
        ----------
        None 
            We use the euclidien distance to compute fitness (we might add a method parameter to test other metrics)

        Returns
        -------
        fitness : list
            List of the fitness of each individual in the same ordre as given by self.population.

        """

        """fitness = []
        for i in range(len(self.population)) :
            # print(self.population[i])
            fitness.append(self.calculate_individual_fitness(self.population[i]))
            # fitness.append(-np.sqrt(np.sum(np.square(self.population[i] - self.target_photo)))) #Compute euclidean distance with the formula"""
        
        # fitness = -np.sqrt(np.sum(np.square(self.population - self.target_photo), axis=1))
        """fitness = -np.linalg.norm(self.population - self.target_photo, axis=1)
        return fitness.tolist()"""
        return fast_norm(self.population - self.target_photo).tolist()
    
    def parallel_calculate_fitness(self):
        fitness = Parallel(n_jobs=-1)(delayed(np.linalg.norm)(indiv - self.target_photo) for indiv in self.population)
        return (-np.array(fitness)).tolist()
    
    def calculate_individual_fitness(self, indiv):
        """
        Function to calculate the fitness of a single individual given as paramter.

        Parameters 
        ----------
        indiv : np.array
            Vector of a given indiviual of the population
        
        Returns 
        -------
        fitness_val : float
            Euclidian distance between the parameter and the target vector

        """ 
        # fitness_val = -np.sqrt(np.sum(np.square(indiv - self.target_photo)))
        fitness_val = -np.linalg.norm(indiv - self.target_photo)
        return fitness_val
    
    def select(self, nb_generation, criteria = "threshold"):

        """
        Select a pair number of individuals according to their fitness. This set of individuals will then 
        endure crossover and mutation.
        
        Possibly use two different criteria to do so : 
            - threshold : choose only the individuals that have a fitness greater than the average one.
            - Fortune_Wheel : include a part of random in the process.
            - tournament : randomization + choose best indiv.

        Parameters
        ----------
        nb_generation : int
            The index of the generation the algorithm is currently in.
        
        criteria : string
            Name of the criteria used for the selection. Default is "threshold".
            Other criterion is "Fortune_Wheel".

        Returns
        -------
        generation : list
            List of individuals selected to pursue algorithm.

        """
        fitness_values = self.dico_fitness[nb_generation] 

        if criteria == "threshold" :
            generation = []
            max_fitness_after_threshold = 0
            pop = None
            
            fitness_avg = np.mean(fitness_values) # Compute average fitness
            for i in range(len(self.population)) : 
                if  self.dico_fitness[nb_generation][i] >= fitness_avg : #Selection of the individuals
                    generation.append(self.population[i])
                else : 
                    if self.dico_fitness[nb_generation][i] > max_fitness_after_threshold :    # Store the individual that is closer to the average fitness in case the number of individuals selected isn't pair.
                        max_fitness_after_threshold, pop = self.dico_fitness[nb_generation][i], self.population[i]

            if len(generation)%2 != 0 : # if we don't have a pair number of individuals
                generation.append(pop)

            return generation

        elif criteria == "Fortune_Wheel" :

            proba_individuals = []
            #table_proba = []
            # max_fitness = np.max(list(self.dico_fitness.values())[-1][:]) # Retrieve the maximum fitness
            
            """elite_size = int(len(self.population) * 0.1)  # Garde les 10 % meilleurs
            sorted_indices = np.argsort(fitness_values)[::-1]
            elite = [self.population[i] for i in sorted_indices[:elite_size]]
            fitness_array = np.array(fitness_values)
            fitness_array = fitness_array[sorted_indices[elite_size:]]"""

            min_fitness = np.min(fitness_values) # compute the most negative fitness -> furtherst away from target
            # transformed_fitness = [abs(f - min_fitness) for f in fitness_values] # translate value so that the closest fitness to 0 has the highest new positive value
            
            transformed_fitness = np.abs(np.array(fitness_values) - min_fitness)
            
            # sum_fitness = sum(transformed_fitness)
            """for i in range(len(self.dico_fitness[nb_generation])): 
                proba_individuals.append(self.dico_fitness[nb_generation][i]*1 / max_fitness) """      # Attribute a probability to each fitness 
                #table_proba.append([self.population[i], self.dico_fitness[nb_generation][i], proba_individuals[i]]) # Création d'une table contenant l'individu, sa fitness et sa probabilité d'être tiré

            """if sum_fitness == 0: # avoid dividing by 0 
                proba_individuals = [1 / len(fitness_values)] * len(fitness_values)  # uniform distribution when they all have nul fitness
            else:
                proba_individuals = [f / sum_fitness for f in transformed_fitness] # normalisation of the fitness value by the sum of all fitness"""

            proba_individuals = transformed_fitness ** 0.5 / np.sum(transformed_fitness ** 0.5) if np.sum(transformed_fitness) != 0 else np.ones_like(transformed_fitness) / len(transformed_fitness)

            if len(self.population)%2 == 0 : # Test la parité de la population pour générer un nombre pair d'individus
                nb_tirages = len(self.population)//2
            else : 
                nb_tirages = len(self.population)//2 + 1
            #sample = choices(table_proba, weights = proba_individuals, k = nb_tirages)  # Tirage de taille_de_population/2 pair individus selon leur probabilité. Attention !! ESt-ce qu'il y a des remises ???
            #generation.append(sample[:][0])

            # nb_tirages -= elite_size  

            ## Ou 
            index_choice = np.random.choice(len(self.population), size = nb_tirages, replace = False, p = proba_individuals) # list of the selected individuals according to their probability
            # index_choice = np.random.choice(sorted_indices[elite_size:], size = nb_tirages, replace = False, p = proba_individuals)
            generation = [self.population[k] for k in index_choice]
            # generation = generation + elite

            """max_indice = np.argsort(fitness_values)[-1]
            elite = self.population[max_indice] 
            generation[-1] = elite"""

            return generation

        elif criteria == "tournament":
            fitness_array = np.array(fitness_values)
            tournament_size = 4
            
            if len(self.population)%2 == 0 : # Test la parité de la population pour générer un nombre pair d'individus
                nb_tirages = len(self.population)//2
            else : 
                nb_tirages = len(self.population)//2 + 1

            selected = []
            for _ in range(nb_tirages): 
                indices = np.random.choice(len(self.population), tournament_size, replace=False) # choose randomly some individuals
                best_idx = indices[np.argmax(fitness_array[indices])]
                selected.append(self.population[best_idx]) # select best individuals in these 

            return selected
        else : 
            print("Error, unknown method")
    
    def crossover_and_mutations(self):
        """
        Last step of each loop of the genetic algorithm : 
        - breeding between parents (crossover) too give a child population, 
        - mutation on these children

        Parameters
        ----------
        crossover_proba : float
            Probability of crossing over between two parents (between 0 and 1)

        Returns
        -------
        new_populations : list
            List of the vectors of the new population composed of the parents 
            (best individuals from the previous generation) and their children.

        """
        new_population = self.generation[:] # new population will also contained the selected parents from the previous generation
        for i in range(0, len(self.generation), 2): # Caution : this imply that the generation contain an even number of parents 
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
            Other methods are "two-points" and "uniform".

        Returns
        -------
        child1 : np.array
            Array of dimension n representing the first child (vector).
        child2 : np.array
            Array of dimension n representing the second child (vector).

        """
        if method == "single-point" : # exchange every coordinates after a random index, including this index.
            crossing_point = np.random.randint(1, self.dimension) # self.dimension = size of the vectors = dimension of the vector space
            # print(crossing_point)
            child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
            child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

        elif method == "two-points" : # exchange every coordinates between two indexes, including them
            low_crossing_point = np.random.randint(0, self.dimension) # lower bound can be any index in the range of the size of the vectors
            upper_bound = min(low_crossing_point + (self.dimension-1), self.dimension) # to be sure that not every coordinates are exchanged (meaning that children = parents), we ensure that at least one stay the same
            high_crossing_point = np.random.randint(low_crossing_point, upper_bound) # upper bound cannot be inferior to the lower one
            # print(low_crossing_point, high_crossing_point)
            child1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
            child2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))

        elif method == "uniform" : # randomly choose from which parents a coordinate will be 
            child1, child2 = np.array([0.0 for _ in range(self.dimension)]), np.array([0.0 for _ in range(self.dimension)])
            """for i in range(self.dimension):
                p = np.random.randint(0,2) # tossing a coin : result = 0 or 1
                if p == 0 : # coordinate of child i come from parent i
                    children1[i] = parent1[i]
                    children2[i] = parent2[i]
                else : # coordinate of child i come from the other parent
                    children1[i] = parent2[i]
                    children2[i] = parent1[i]"""
            mask = np.random.randint(0, 2, size=self.dimension, dtype=bool)
            child1, child2 = np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)

            if np.array_equal(child1, parent1) : # if no exchange were made, while we want at least one when there is a crossover:
                random_index = np.random.randint(self.dimension) # randomly choose a coordinate to exchange to ensure that children differ from parents
                child1[random_index] = parent2[random_index]
                child2[random_index] = parent1[random_index]
        
        elif method == "BLX-alpha":
            alpha = 0.5
            lower = np.minimum(parent1, parent2) - alpha * np.abs(parent1 - parent2)
            upper = np.maximum(parent1, parent2) + alpha * np.abs(parent1 - parent2)
            child1 = np.random.uniform(lower, upper)
            child2 = np.random.uniform(lower, upper)

        elif method == "max_diversity": 
            alpha = np.random.uniform(0.3, 0.7)  # Varie aléatoirement l’intensité du mélange
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            # T = max(0.2, 1 - self.count_generation / self.max_iteration)
            return child1 + np.random.normal(0, 0.2, size=parent1.shape), child2 + np.random.normal(0, 0.2, size=parent2.shape)

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
            meaning that mutating a lot of gene increases the chances of moving away from it. On the contrary,
            a badly fitted chromosome has all interest in having a high mutation probability to change a lot 
            of its coordinate and move closer to the target. Therefore, it can be useful to have two different 
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
            Standard deviation of the normal distribution from whcih are drawn mutations.
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
                """if fit_chr >= -30 :
                    sigma_mutation /= 5 # smaller mutation when closer to target"""
            else : # individual is badly fitted compared to the rest of the population
                proba_mutation = self.mutation_rate[0]
                # sigma_mutation *= 1.2 # higher size of mutation for individuals with a bad fitness

        else : 
            print("Error, unknow method") # if the method is incorrect, do not apply any mutation 
            return 

        """for i in range(self.dimension):
            if np.random.random_sample() < proba_mutation : 
                m = np.random.normal(0, sigma_mutation) # mutations are drawn from a normal distribution distributed around 0. Modifying sigma allows to have bigger or smaller mutations
                chr[i] += m"""
        
        T = max(0.01, 1 - self.count_generation / self.max_iteration)
        """if np.std(self.population, axis=0).mean() < 0.1:  # Seuil de diversité
            self.sigma_mutation *= 1.2  # Réaugmente la mutation temporairement"""
        mask = np.random.rand(self.dimension) < proba_mutation
        mutations = np.random.normal(0, sigma_mutation*T, self.dimension)
        chr[mask] += mutations[mask]

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
        index_generations = list(range(1, list(self.dico_fitness.keys())[-1]+1))
        # print(list(self.dico_fitness.values())[-1])
        best_fitness_values = [np.max(list_fitness) for list_fitness in self.dico_fitness.values()]
        worst_fitness_values = [np.min(list_fitness) for list_fitness in self.dico_fitness.values()]

        plt.figure()
        plt.plot(index_generations, best_fitness_values, label='Best Fitness', color='black')
        plt.fill_between(index_generations, worst_fitness_values, best_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()
        plt.show()

    def stop_condition(self):
        """
        Stopping condition of the GA : 
        If the m individuals of the population with the highest fitness values 
        all have a fitness inferior to an arbitrary threshold, stop the algorithms :
        we consider that we are close enough from the target.

        Parameters 
        ----------
        m : int
            Number of vectors to retrieve
        threshold : float
            Max distance until when we consider that we are close from the target

        Returns
        -------
        stop : bool
            True or false, depending on if we need to stop the GA or not.

        """
        self.solution, max_fitness = self.retrieve_max_fitness_population() #  to retrieve only the m closest individuals to the target

        stop = True
        for val in max_fitness :
            if val < self.threshold :
                stop = False
                break
    
        return stop
    
    def retrieve_max_fitness_population(self):
        """
        Retrieve the m more fitted solutions in the population, 
        which are the m closest vectors to the target.

        Parameters
        ----------
        m : int 
            Number of vectors to retrieve

        Returns
        -------
        population_max : list
            List of the m vectors having the best fitnesses
        fitness_max : np.array
            Array of the m highest fitness values

        """
        last_fitnesses = list(self.dico_fitness.values())[-1][:]
        """last_fitnesses.sort()
        print(last_fitnesses[-m:])""" # only retrieve max fitness but not their index in the list

        indices_max = np.argsort(last_fitnesses)[-self.m:][::-1] # finding the indices corresponding to the m highest values, in decreasing order
        fitness_max = np.sort(last_fitnesses)[-self.m:][::-1] # m highest values of fitness (used in the stop condition)
        population_max = [self.population[i] for i in indices_max]
        return population_max, fitness_max
    
    def main_loop(self):
        """
        Main Loop of the Genetic Algorithm.
        The loop represent the evolution of the population over generations.

        Parameters
        ----------
        None

        Returns
        -------
        self.solution : list
            List of the vectors solution of the Genetic Algorithm
        """
        
        while self.count_generation < self.max_iteration :
            self.count_generation += 1 
            # print(self.count_generation)
            self.dico_fitness[self.count_generation] = self.calculate_fitness()
            if self.stop_condition(): # if we have solutions close enough to the target
                break
            self.generation = self.select(self.count_generation, criteria=self.selection_method) 
            new_population = self.crossover_and_mutations()
            self.population = new_population

            """if self.count_generation % 50 == 0:
                self.refine_best_individual()"""
            """if self.count_generation % 50 == 0:  # Toutes les 50 générations
                n_migrants = max(1, len(self.population) // 10)  # Remplace 10% de la pop
                self.population[:n_migrants] = np.random.uniform(-10, 10, (n_migrants, self.dimension))"""

            # self.sigma_mutation *= 0.98 # decrease the size of mutations at each generation because closer to the target, smaller mutations are more beneficial
           
        # print(np.max(self.dico_fitness[self.count_generation]))
        # print("Écart-type des coordonnées finales :", np.std(self.solution, axis=0).mean())
        # self.solution = self.retrieve_final_population(10) # already done in self.stop_condition()
        return self.solution
    
    def refine_best_individual(self):
        best_idx = np.argmax(self.dico_fitness[self.count_generation])
        best_individual = self.population[best_idx]

        result = minimize(lambda x: np.linalg.norm(x - self.target_photo), best_individual, method="BFGS")
        
        if result.success:
            self.population[best_idx] = result.x  # Remplace par la version optimisée



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
    print("\n")

def test_mutation(ga, method = "constant"):
    chr = np.array([1,4,10,2,1,0,0,5,22,1,3,16,0,1,7,0], dtype = float)
    print("Before mutation: ")
    print(chr) 
    ga.dico_fitness[ga.count_generation] = ga.calculate_fitness()
    ga.mutation(chr, 1, method)
    print("After mutation: ")
    print(chr)
    print("\n")

def test_create_random_init_pop(ga, target) : 
    print("Target :")
    print(target)
    print("Initial population :")
    print(ga.create_random_init_pop(10))

def test_calculate_fitness(ga) : 
    print("Fitness : ")
    print(ga.calculate_fitness())


def test_unitaire():
    target = np.array([1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,0])
    ga = GeneticAlgorithm(target, max_iteration=1000, size_pop=100, nb_to_retrieve=10, stop_threshold=-10, selection_method="Fortune_Wheel",
                          crossover_proba=0.7, crossover_method="uniform", mutation_rate=(0.5, 0.05), sigma_mutation=1.5, mutation_method="adaptive")
    test_crossing_over(ga, "uniform")
    test_mutation(ga, "adaptive")
    test_create_random_init_pop(ga, target)
    test_calculate_fitness(ga)

def test_global():
    target = np.random.rand(8,8,128) * 10
    target = target.flatten(order = "C")
    print(f"target : {target}")
    ga = GeneticAlgorithm(target, max_iteration=500, size_pop=100, nb_to_retrieve=10, stop_threshold=-10, selection_method="Fortune_Wheel",
                          crossover_proba=0.9, crossover_method="max_diversity", mutation_rate=(0.5, 0.05), sigma_mutation=0.5, mutation_method="adaptive")
    solutions = ga.main_loop()
    ga.visualization()
    print(len(solutions))
    for v in solutions :
        print(v)

def test_separation():
    target = np.random.rand(8,8,4) * 10
    dimensions = target.shape
    solutions = []
    for i in range(dimensions[2]): # separation of the different canal of the vector
        partial_target = target[:,:,i].flatten(order = "C")
        ga = GeneticAlgorithm(partial_target, max_iteration=500, size_pop=100, nb_to_retrieve=10, stop_threshold=-10, selection_method="Fortune_Wheel",
                          crossover_proba=0.9, crossover_method="max_diversity", mutation_rate=(0.5, 0.05), sigma_mutation=0.5, mutation_method="adaptive")
        partial_solutions = ga.main_loop()
        ga.visualization()
        solutions.append(partial_solutions)
    

    # reconstruction of vectors of the good size
    reconstructed_solutions = []
    
    for j in range(len(solutions[0])):
        reconstruction = np.concatenate([solutions[k][j] for k in range(len(solutions))])
        reconstruction = np.reshape(reconstruction, target.shape, order = "C")
        
        # print(reconstruction.shape)
        reconstructed_solutions.append(reconstruction)
        

    print(f"target : {target}")
    print(f"solutions :")
    for s in reconstructed_solutions:
        print(s)

    print("Écart-type des coordonnées finales :", np.std(reconstructed_solutions).mean())

    for s in reconstructed_solutions: 
        print(f" norm : {-np.linalg.norm(s-target)}")


def run_ga(i, target):
    partial_target = target[i, :, :].flatten(order="C")
    ga = GeneticAlgorithm(partial_target, max_iteration=1000, size_pop=100, nb_to_retrieve=10, stop_threshold=-5, 
                            selection_method="Fortune_Wheel", crossover_proba=0.9, crossover_method="max_diversity", 
                            mutation_rate=(0.5, 0.05), sigma_mutation=0.8, mutation_method="adaptive")
    return ga.main_loop()

def real_separation(target):
    """
    target : numpy array
    reconstructed_solutions : numpy array
    """
    dimensions = target.shape
    solutions = Parallel(n_jobs=-1)(delayed(run_ga)(i, target) for i in range(dimensions[0]))

    reconstructed_solutions = np.stack(solutions, axis=1).reshape((-1, *target.shape), order="C")
        
    """print(f"target : {target}")
    print(f"solutions :")
    for s in reconstructed_solutions:
        print(s)

    print("Écart-type des coordonnées finales :", np.std(reconstructed_solutions).mean())

    for s in reconstructed_solutions: 
        print(f" norm : {-np.linalg.norm(s-target)}")"""
    
    return reconstructed_solutions

if __name__ == "__main__" :
    # test_unitaire()
    # test_global()
    # test_separation()
    target = np.random.rand(128,8,8) * 10
    real_separation(target)


    
