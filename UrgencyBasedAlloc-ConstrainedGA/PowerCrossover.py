import copy
from PowerIndividual import PowerIndividual
from My_GA.Operators.MyCrossover import MyCrossover
from util import power_adjustment, power_adjustment_without_sort
import numpy as np


class PowerCrossover(MyCrossover):
    def __init__(self, rate, para_manager, alpha=None):
        super().__init__(rate, para_manager, alpha)
        self._individual_class = [PowerIndividual]

    @staticmethod
    def cross_individuals(individual_a, individual_b, pos, alpha):
        '''
        generate two individuals based on parent individuals:
            - individual_a, individual_b: the selected individuals
            - pos  : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: two generated individuals
        '''
        # 二进制和实数交叉
        seq_pos_p = pos[-individual_a.dimension:]
        solution_a = individual_a.solution.copy()
        solution_b = individual_b.solution.copy()

        seq_pos_p = np.reshape(seq_pos_p, (len(seq_pos_p), 1))
        seq_pos_p_nbits = np.reshape(np.concatenate([seq_pos_p] * individual_a.power_bit, axis=1),
                                   individual_a.dimension * individual_a.power_bit)
        seq_p_temp = solution_a[seq_pos_p_nbits].copy()
        solution_a[seq_pos_p_nbits] = solution_b[seq_pos_p_nbits].copy()
        solution_b[seq_pos_p_nbits] = seq_p_temp

        # power_adjustment(individual_a)
        # power_adjustment(individual_b)
        # power_adjustment_without_sort(individual_a)
        # power_adjustment_without_sort(individual_b)

        # return new individuals
        new_individual_a = individual_a.__class__(individual_a.ranges, individual_a.dimension,
                                                  individual_a.para_manager, individual_a.assignment_channel_policy)
        new_individual_b = individual_b.__class__(individual_b.ranges, individual_b.dimension,
                                                  individual_b.para_manager, individual_b.assignment_channel_policy)

        new_individual_a.solution = solution_a
        new_individual_a.init_evaluation()
        new_individual_a.fitness = copy.deepcopy(new_individual_a.evaluation)
        new_individual_b.solution = solution_b
        new_individual_b.init_evaluation()
        new_individual_b.fitness = copy.deepcopy(new_individual_b.evaluation)

        return new_individual_a, new_individual_b

    def cross(self, population):
        '''
        population: population to be crossed. population should be evaluated in advance
                    since the crossover may be based on individual fitness.
        '''
        self.cross_num = 0
        # 排序配对
        evaluation = np.array([I.evaluation for I in population.individuals])
        pos = np.argsort(evaluation)
        individuals_a = []
        individuals_b = []
        # 排序组合
        for i in range(int(population.size/2)):
            individuals_a.append(population.individuals[pos[i]])
            individuals_b.append(population.individuals[pos[i+int(population.size/2)]])
        new_individuals, count = [], 0
        for individual_a, individual_b in zip(individuals_a, individuals_b):
            # crossover
            if np.random.rand() <= self._adaptive_rate(individual_a, individual_b, population):
                self.cross_num += 1
                # random position to cross
                pos = self._cross_positions(individual_a.dimension)
                child_individuals = self.cross_individuals(individual_a, individual_b, pos, self._alpha)
                new_individuals.extend(child_individuals)

            # skip crossover, but copy parents directly
            else:
                new_individuals.append(copy.deepcopy(individual_a))
                new_individuals.append(copy.deepcopy(individual_b))

            # generate two child at one crossover
            count += 2

            # stop when reach the population size
            if count > population.size:
                break

        # the count of new individuals may lower than the population size
        # since same parent individuals for crossover would be ignored
        # so when count < size, param `replace` for choice() is True,
        # which means dupilcated individuals are necessary
        return new_individuals
