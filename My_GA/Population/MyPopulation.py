import copy

from GA.GAPopulation.Population import Population
from My_GA.util.HyperParameter import ParameterManager
from My_GA.Population.SeqReaIndividual import SeqReaIndividual
import numpy as np


class MyPopulation(Population):
    def __init__(self, individual: SeqReaIndividual, size, preservation_size, para_manager: ParameterManager):
        super().__init__(individual, size)
        self.preservation_individuals = []
        self.preservation_size = preservation_size
        self.para_manager = para_manager

    def initialize(self):
        IndvClass = self.individual.__class__
        self.individuals = np.array([IndvClass(self.individual.ranges, self.individual.dimension, self.para_manager) for i in range(self.size)], dtype=IndvClass)

    def preserve_elitism(self):
        if len(self.preservation_individuals) >= self.para_manager.PreservationSize:
            worst_pos = np.argsort([I.evaluation for I in self.preservation_individuals])[0]
            if self.best.evaluation>self.preservation_individuals[worst_pos].evaluation:
                self.preservation_individuals[worst_pos] = self.best
        else:
            self.preservation_individuals.append(self.best)

    @property
    def best(self):
        '''get best individual according to evaluation value'''
        # collect evaluations
        evaluation = np.array([I.evaluation for I in self.individuals])

        # get the maximum position
        pos = np.argmax(evaluation)
        return copy.deepcopy(self.individuals[pos])
