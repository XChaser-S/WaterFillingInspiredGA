import copy
from My_GA.Population.SeqReaIndividual import SeqReaIndividual
from GA.GAOperators.Operators import Crossover
import numpy as np


class MyCrossover(Crossover):
    def __init__(self, rate, para_manager, alpha=None):
        super().__init__(rate, alpha)
        self.para_manager = para_manager
        self._individual_class = [SeqReaIndividual]
        self.cross_num = 0

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
        seq_pos = pos[:individual_a.dimension[0]]
        real_pos = pos[-individual_a.dimension[0]:]
        solution_a = individual_a.solution.copy()
        solution_b = individual_b.solution.copy()

        seq_pos = np.reshape(seq_pos, (len(seq_pos), 1))
        seq_pos_nbits = np.reshape(np.concatenate([seq_pos]*individual_a.num_bit, axis=1),
                                   individual_a.dimension[0]*individual_a.num_bit)
        seq_temp = solution_a[0][seq_pos_nbits].copy()
        solution_a[0][seq_pos_nbits] = solution_b[0][seq_pos_nbits].copy()
        solution_b[0][seq_pos_nbits] = seq_temp

        real_exchange_a = solution_a[1][real_pos].copy()
        real_exchange_b = solution_b[1][real_pos].copy()
        rand_a = np.random.random()
        rand_b = np.random.random()
        solution_a[1][real_pos] = rand_a*real_exchange_a + (1-rand_a)*real_exchange_b
        solution_b[1][real_pos] = rand_b*real_exchange_b + (1-rand_b)*real_exchange_a

        # return new individuals
        new_individual_a = individual_a.__class__(individual_a.ranges, individual_a.dimension, individual_a.para_manager)
        new_individual_b = individual_b.__class__(individual_b.ranges, individual_b.dimension, individual_b.para_manager)

        new_individual_a.solution = solution_a
        new_individual_a.init_evaluation()
        new_individual_a.fitness = copy.deepcopy(new_individual_a.evaluation)
        new_individual_b.solution = solution_b
        new_individual_b.init_evaluation()
        new_individual_b.fitness = copy.deepcopy(new_individual_b.evaluation)

        return new_individual_a, new_individual_b

    @staticmethod
    def _cross_positions(dimension):
        '''generate a random and continuous range of positions for crossover'''
        # 随机产生交叉掩膜
        # start, end position
        pos = np.random.randint(0, 2, dimension, np.bool_)
        return pos

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
                pos = self._cross_positions(individual_a.dimension[0]+individual_a.dimension[1])
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
