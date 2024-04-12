from GA.GAOperators.Operators import Selection
from My_GA.Population.MyPopulation import MyPopulation
import numpy as np
import copy


class LinearRankingSelection(Selection):
    def select(self, population: MyPopulation):
        size = population.size + len(population.preservation_individuals)
        p_min = 2/(size*(size+1))
        p_max = 2/(size+1)
        evaluation = np.array([I.evaluation for I in population.individuals] +
                              [I.evaluation for I in population.preservation_individuals])
        pos = np.argsort(evaluation)
        selection_probability = p_min + (p_max-p_min)*np.arange(size)/(size-1)

        selected_individuals = np.random.choice(np.array(list(population.individuals)+
                                                         list(population.preservation_individuals))[pos],
                                                population.size, p=selection_probability)

        return np.array([copy.deepcopy(I) for I in selected_individuals])
