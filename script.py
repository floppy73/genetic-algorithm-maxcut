import ioh
import os
import numpy as np
import matplotlib.pyplot as plt

class RandomSearch:
    def __call__(self, problem: ioh.ProblemType):
        x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(x)

class YourAlgorithm:
    
    def __init__(self, population_size = 30, generations = 300, crossover_rate = 0.8, mutation_rate = 0.001, tournament_size = 3):
        self.generations = generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def _choice(self, populations, fitnesses):
        #use tournament choice
        selected_indexes = []
        for _ in range(2):
            candidates = np.random.choice(np.arange(self.population_size), size=self.tournament_size, replace=False)
            tournament_fitnesses = fitnesses[candidates]
            selected_index = candidates[np.argmax(tournament_fitnesses)]
            selected_indexes.append(selected_index)
        return populations[selected_indexes]


    def _cross_over(self, parent1, parent2):

        #use binomial crossover
        if np.random.rand() > self.crossover_rate:
            child1 = parent1
            child2 = parent2
        else:
            mask = np.random.randint(0, 2, size = len(parent1)).astype(bool)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def _mutation(self, offsprings):
        for offspring in offsprings:
            for i in range(len(offspring)):
                if np.random.rand() < self.mutation_rate:
                    offspring[i] = 1 - offspring[i]
        return offsprings
    
    def __call__(self, problem: ioh.ProblemType):
        """You should implement search behaviour here"""
        populations = np.array([np.random.randint(0, 2, size=problem.meta_data.n_variables) for _ in range(self.population_size)])
        fitnesses = np.array([problem(populations[x]) for x in range(self.population_size)])

        offsprings = []

        for _ in range(self.generations):
            #generate offspring until the number of them attain the initial population size
            while len(offsprings) < self.population_size:
                parents = self._choice(populations, fitnesses)
                child1, child2 = self._cross_over(parents[0], parents[1])
                offsprings.append(child1)
                offsprings.append(child2)
            offsprings = self._mutation(offsprings)
            #generational change 
            populations = np.array(offsprings.copy())
            fitnesses = np.array([problem(populations[x]) for x in range(self.population_size)])
            offsprings = []


if __name__ == "__main__":
    problems = [
        ioh.get_problem(pid, problem_class=ioh.ProblemClass.GRAPH)
        for pid in [2000, 2001, 2002, 2003, 2004]
    ]

    tournament_size = 20
    n_runs = 10
    
    for alg in (RandomSearch, YourAlgorithm):
        name = alg.__name__
        #instantiate a logger
        logger = ioh.logger.Analyzer(algorithm_name=name + "size" + str(tournament_size), folder_name=name + "_size" + str(tournament_size))

        for problem in problems:
            problem.attach_logger(logger)
            for run in range(n_runs):
                optimizer = alg() if name == "RandomSearch" else alg(tournament_size=tournament_size)
                optimizer(problem)
                problem.reset()

        logger.close()
        
    