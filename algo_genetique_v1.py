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
    1.1 (07/03/2025)
"""
import numpy as np
import matplotlib.pyplot as plt
from random import choices

class GeneticAlgorithm():
    """
    Class GeneticAlgorithm

    This algorithm aims at creating a set of images similar to a target picture using 
    a genetic algorithm. 
    These images are all representing as vectors of dimension n, which are the 
    representation of the images in the latent space of an autoencoder.

    """
    def __init__(self, target, max_iteration):
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

        self.population = self.create_random_init_pop(10) # list : initial population from which the evolutionary process begins
        self.generation = None # list : selected population based on fitness at each generation
         
        # self.count_generation = 1 # to count the number of generations and plot it afterwards. # pas utile si génération en clé du dico

        self.dico_fitness = {} # dictionnary to memorize the fitness values of the population at each generation

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
        init_population : list
            List of the vectors of the initial individuals generated randomly

        """

        init_population = [] # normalement pas besoin de init_pop : just faire self.population.append(...)

        for _ in range(size_pop) : 
            # Attention la vrai population ça sera pas juste des entiers et surtout pas que des 0 et des 1
            init_population.append(np.random.randint(2, size = self.dimension)) #Generate an individual randomly

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

        fitness = []

        for i in range(len(self.population)) :
            fitness.append(np.sqrt(np.sum(np.square(self.population[i] - self.target_photo)))) #Compute euclidean distance with the formula

        return fitness
    
    def select(self, nb_generation, criteria = "threshold"):

        """
        Select a pair number of individuals according to their fitness. This set of individuals will then 
        endure crossover and mutation.
        
        Possibly use two different criteria to do so : 
            - threshold : choose only the individuals that have a fitness greater than the average one.
            - Roulette_Wheel : include a part of random in the process.

        Parameters
        ----------
        nb_generation : int
            The index of the generation the algorithm is currently in.
        
        criteria : string
            Name of the criteria used for the selection. Default is "threshold".
            Other criterion is "Roulette_Wheel".

        Returns
        -------
        generation : list
            List of individuals selected to pursue algorithm.

        """
        generation = []

        if criteria == "threshold" :
            max_fitness_after_threshold, pop = 0
            fitness_avg = sum(self.dico_fitness[nb_generation])/len(self.dico_fitness[nb_generation]) # Compute average fitness
            for i in range(len(self.population)) : 
                if  self.dico_fitness[nb_generation][i] >= fitness_avg : #Selection of the individuals
                    generation.append(self.population[i])
                else : 
                    if self.dico_fitness[nb_generation][i] > max_fitness_after_threshold :    # Store the individual that is closer to the average fitness in case the number of individuals semected isn't pair.
                        max_fitness_after_threshold, pop = self.dico_fitness[nb_generation][i], self.population[i]

            if len(generation)%2 != 0 :       # if we don't have a pair number of individuals
                generation.append(pop) 

        elif criteria == "Roulette_Wheel" :
            # A faire 
            proba_individuals = []
            list_proba = []
            print(self.dico_fitness.values())
            print((list(self.dico_fitness.values())[-1][:]))
            max_fitness = np.argmax(list(self.dico_fitness.values())[-1][:]) # Retrieve the maximum fitness 
            for i in range(len(self.dico_fitness)): 
                proba_individuals.append(self.dico_fitness[nb_generation][i]*1 / max_fitness)       # Attribute a probability to each fitness 
                list_proba.append([self.population[i], self.dico_fitness[nb_generation][i], proba_individuals[i]]) # Création d'une table contenant l'individu, sa fitness et sa probabilité d'être tiré
            
            if len(self.population)%2 == 0 :          # Test la parité de la population pou générer un nombre pair d'individus
                nb_tirages = len(self.population)/2
            else : 
                nb_tirages = len(self.population)/2 + 1
            sample = choices(list_proba, weights = proba_individuals, k = nb_tirages)  # Tirage de taille_de_population/2 pair individus selon leur probabilité.
            generation.append(sample[:][0])

        else : 
            print("Error, unknown method")

        return generation
    
    def crossover_and_mutations(self, crossover_proba):
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
            if np.random.random_sample < crossover_proba :
                child1, child2 = self.crossover(parent1, parent2, method="single-point")
            else : # no crossing over : children are equal to parents (before mutation)
                child1, child2 = parent1, parent2
            
            # Mutations
            new_population.append(self.mutation(child1), 0.1, 0.5, method="constant")
            new_population.append(self.mutation(child2), 0.1, 0.5, method="constant")
            
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
    
    def mutation(self, chr, mutation_rate, sigma_mutation, method = "constant"):
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
            If method is "adaptive", mutation rate is a tuple of two floats.
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
            proba_mutation = mutation_rate
            
        elif method == "adaptive": # in the adaptive case, the mutation rate depends on the fitness of the chromosome
            fit_avg = np.mean([self.calculate_fitness(indiv) for indiv in self.generation]) # average fitness among the population
            fit_chr = self.calculate_fitness(chr) # fitness of the current individual
            if fit_chr >= fit_avg : # individual is well fitted compared to the rest of the population 
                proba_mutation = mutation_rate[1]
            else : # individual is badly fitted compared to the rest of the population
                proba_mutation = mutation_rate[0]

        else : 
            print("Error, unknow method") # if the method is incorrect, do not apply any mutation 
            return 

        for i in range(self.dimension):
            if np.random.random_sample() < proba_mutation : 
                m = np.random.normal(0, sigma_mutation) # mutations are drawn from a normal distribution distributed around 0. Modifying sigma allows to have bigger or smaller mutations
                chr[i] += m

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
        index_generations = range(1, list(self.dico_fitness.keys)[-1])
        best_fitness_values = [[max(fitness) for fitness in list_fitness] for list_fitness in self.dico_fitness.values()]
        worst_fitness_values = [[min(fitness) for fitness in list_fitness] for list_fitness in self.dico_fitness.values()]

        plt.figure()
        plt.plot(index_generations, best_fitness_values, label='Best Fitness', color='black')
        plt.fill_between(index_generations, worst_fitness_values, best_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()

    def stop_condition(self):
        return
    
    def retrieve_final_population(self, m):
        """
        Retrieve the m more fitted solutions in the population, 
        which are the m closest vectors to the target.

        Parameters
        ----------
        m : int 
            Number of vectors to retrieve

        Returns
        -------
        solutions : list
            List of the m vectors having the best fitnesses

        """
        last_fitnesses = list(self.dico_fitness.values())[-1][:]
        """last_fitnesses.sort()
        print(last_fitnesses[-m:])""" # only retrieve max fitness but not their index in the list

        list_max_index = []
        for _ in range(m):
            next_index = np.argmax(last_fitnesses)
            list_max_index.append(next_index)
            last_fitnesses.pop(next_index)
        # print(list_max_index)

        array_pop = np.array(self.population)
        solutions = list(array_pop[list_max_index])
        return solutions
    
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
        count_generation = 0
        while count_generation < self.max_iteration :
            count_generation += 1 
            self.dico_fitness[count_generation] = self.calculate_fitness()
            if self.stop_condition(): # if we have solutions close enough to the target
                break
            self.generation = self.select(count_generation) 
            new_population = self.crossover_and_mutations(crossover_proba=0.7)
            self.population = new_population

        self.solution = self.retrieve_final_population(10) # to retrieve only the m closest individuals to the target
        self.visualization()
        return self.solution



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
    ga.mutation(chr, 0.5, 1)
    print("After mutation: ")
    print(chr)
    print("\n")

def test_create_random_init_pop (ga, target) : 
    print("Target :")
    print(target)
    print("Initial population :")
    print(ga.create_random_init_pop(10))

def test_calculate_fitness(ga) : 
    print("Fitness : ")
    print(ga.calculate_fitness())

if __name__ == "__main__" :
    target = np.array([1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,0])
    ga = GeneticAlgorithm(target, 10000)
    test_crossing_over(ga, "uniform")
    test_mutation(ga)
    test_create_random_init_pop(ga, target)
    test_calculate_fitness(ga)
    
