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
    4.1 (31/03/2025)
"""

#### Libraries ####
import numpy as np
import matplotlib.pyplot as plt
from random import choices

from numba import njit
from joblib import Parallel, delayed
from scipy.optimize import minimize

import time

class GeneticAlgorithm():
    
    def __init__(self, picture1, picture2, nb_to_retrieve, crossover_method, mutation_rate, sigma_mutation):
        self.p1 = picture1
        self.p2 = picture2

        self.dimension = len(self.p1) # dimension of the vector space = "number of gene of one individual"

        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.sigma_mutation = sigma_mutation
        
        self.solutions = self.crossover_and_mutations(nb_to_retrieve, self.dimension//5)

    def crossover_and_mutations(self, nb_to_retrieve, size_crossover = None):
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
        # Crossovers
        nb_iteration = nb_to_retrieve//2 # nb_to_retrieve must be even

        arr_crossing_points = np.linspace(0.2, 0.8, nb_iteration)*self.dimension
        arr_crossing_points = np.rint(arr_crossing_points).astype(np.int64)
    
        if self.crossover_method == "two-points" :
            arr_lower_points = np.linspace(0.25, 0.75, nb_iteration)*self.dimension
            arr_lower_points = np.rint(arr_lower_points).astype(np.int64)
            arr_upper_points = arr_lower_points + size_crossover
            arr_crossing_points = np.array([(arr_lower_points[i], arr_upper_points[i]) for i in range(len(arr_lower_points))])

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
            self.mutation(child1)
            self.mutation(child2)

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
        if self.crossover_method == "single-point" : # exchange every coordinates after a random index, including this index.
            child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
            child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

        elif self.crossover_method == "two-points" : # exchange every coordinates between two indexes, including them
            low_crossing_point = crossing_point[0] # lower bound can be any index in the range of the size of the vectors
            high_crossing_point = crossing_point[1] # upper bound cannot be inferior to the lower one
            
            child1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
            child2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))

        elif self.crossover_method == "uniform" : # randomly choose from which parents a coordinate will be 
            child1, child2 = np.array([0.0 for _ in range(self.dimension)]), np.array([0.0 for _ in range(self.dimension)])
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
        Implementation of a mutational event on a chromosome/individual (vector).
        Each gene (coordinate) of the chromosome as a probability of mutating depending
        on the mutation rate given in parameter. 

        Mutations are drawn from a normal distribution distributed around 0. 
        Modifying the standard deviation sigma allows to have bigger or smaller mutations.

        Parameters
        ----------
        chr : 
        mutation_rate : float or tuple
            probability of mutation of a gene (between 0 and 1).
            If method is "constant", mutation rate is a unique float.
            If method is "adaptive", mutation rate is a tuple of two floats : (rate_for_low_fitness, rate_for_high_fitness)
        sigma_mutation : float
            Standard deviation of the normal distribution from which are drawn mutations.

        Returns
        -------
        None
            chr is modified directly by the function and thus do not need to be returned.

        """
        mask = np.random.rand(self.dimension) < self.mutation_rate # only mutate with a certain probability
        mutations = np.random.normal(0, self.sigma_mutation, self.dimension)
        chr[mask] += mutations[mask] # add random component to the vectors when its coordinates have mutated.

def test_mutation(picture1, picture2):
    ga = GeneticAlgorithm(picture1, picture2, nb_to_retrieve=6, crossover_method="single-point", 
                          mutation_rate=0.5, sigma_mutation=1)
    
    chr = np.random.rand(4,4) * 20
    chr = chr.flatten(order="C")
    print("Before mutation: ")
    print(chr) 
    ga.mutation(chr)
    print("After mutation: ")
    print(chr)
    print("\n")

def test_crossover(picture1, picture2, point_of_crossover, w, alpha, method):
    ga = GeneticAlgorithm(picture1, picture2, nb_to_retrieve=6, crossover_method=method, 
                          mutation_rate=0.5, sigma_mutation=1)
    
    print("---- Parents ----")
    print(picture1)
    print(picture2)
    child1, child2 = ga.crossover(picture1, picture2, point_of_crossover, w, alpha)
    print("---- Children ----")
    print(child1)
    print(child2)

def run_ga(targets, nb_solutions, crossover_method, mutation_rate, sigma_mutation):
    dimensions = targets[0].shape
    flat_targets = [t.flatten(order="C") for t in targets]

    ga = GeneticAlgorithm(flat_targets[0], flat_targets[1], nb_to_retrieve=nb_solutions, crossover_method=crossover_method, 
                          mutation_rate=mutation_rate, sigma_mutation=sigma_mutation)
    
    solutions = ga.solutions
    solutions = solutions.reshape((-1, *dimensions), order="C") 
    return solutions

if __name__ == "__main__" :
    print(__name__)
    target = np.random.rand(4,4) * 20
    target2 = np.random.rand(4,4) * 20
    # test_mutation(target.flatten(order="C"), target2.flatten(order="C"))
    # test_crossover(target.flatten(order="C"), target2.flatten(order="C"), (8, 12), 0.5, 0.3, "blending")

    solutions = run_ga([target, target2], nb_solutions=6, crossover_method="blending", mutation_rate=0.5, sigma_mutation=1)


    print("---- Parents ----")
    print(target)
    print(target2)
    print("---- Children ----")
    for s in solutions :
        print(s)




    
