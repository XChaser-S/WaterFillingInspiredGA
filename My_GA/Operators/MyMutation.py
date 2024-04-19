import copy
from GA.GAOperators.Operators import Mutation
from My_GA.Population.SeqReaIndividual import SeqIndividual
from My_GA.util.HyperParameter import ParameterManager
from My_GA.util.util import power_adjustment, gray2decimal, decimal2gray
import numpy as np


class MyMutation(Mutation):
    def __init__(self, rate, para_manager: ParameterManager):
        super().__init__(rate)
        self._individual_class = [SeqIndividual]
        self.para_manager = para_manager
        self.mutation_num = 0

    @property
    def individual_class(self):
        return self._individual_class

    @staticmethod
    def mutate_individual(individual: SeqIndividual, positions, alpha):
        '''
        get mutated solution based on the selected individual:
            - individual: the selected individual
            - positions : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: the mutated solution
        '''
        # 设计掩膜
        solution = individual.solution.copy()
        seq_pos_mask = positions[0][:individual.dimension[0] * individual.num_bit]
        seq_pos_p_mask = positions[0][-individual.dimension[1] * individual.power_bit:]
        seq_pos_index = positions[1]
        seq_pos_p_index = positions[2]-(individual.dimension[0] * individual.num_bit)

        # 取反
        solution[0][seq_pos_mask] = ((~solution[0][seq_pos_mask].astype(np.int32)) + 2).astype(np.str_)
        solution[1][seq_pos_p_mask] = ((~solution[1][seq_pos_p_mask].astype(np.int32)) + 2).astype(np.str_)
        # fix the illegal mutation
        for pos in (seq_pos_index / individual.num_bit).astype(np.int32):
            # PL_index = ''
            # for i in solution[0][pos*individual.num_bit:(pos+1)*individual.num_bit]:
            #     PL_index += i
            # PL_index = int(PL_index, 2)
            # if list(solution[0][pos*individual.num_bit:(pos+1)*individual.num_bit]) == list('100'):
            #     print()
            PL_index = gray2decimal(solution[0][pos*individual.num_bit:(pos+1)*individual.num_bit])
            if PL_index > individual.para_manager.NumPL:
                # solution[0][pos * individual.num_bit:(pos + 1) * individual.num_bit] = \
                #     list(bin(individual.para_manager.NumPL)[2:].zfill(individual.num_bit))
                # if decimal2gray(individual.para_manager.NumPL, individual.num_bit) == '111':
                #     print(pos)
                PL_index = np.random.choice(individual.para_manager.NumPL)+1
                solution[0][pos * individual.num_bit:(pos + 1) * individual.num_bit] = \
                    list(decimal2gray(PL_index, individual.num_bit))

        for pos in (seq_pos_p_index / individual.power_bit).astype(np.int32):
            power_level = gray2decimal(solution[1][pos*individual.power_bit:(pos+1)*individual.power_bit])
            if power_level > individual.para_manager.PowerLevels:
                # solution[0][pos * individual.num_bit:(pos + 1) * individual.num_bit] = \
                #     list(bin(individual.para_manager.NumPL)[2:].zfill(individual.num_bit))
                # if decimal2gray(individual.para_manager.NumPL, individual.num_bit) == '111':
                #     print(pos)
                power_level = np.random.choice(individual.para_manager.PowerLevels)+1
                solution[1][pos * individual.power_bit:(pos + 1) * individual.power_bit] = \
                    list(decimal2gray(power_level, individual.power_bit))
        power_adjustment(individual)

        return copy.deepcopy(solution)

    def _adaptive_rate(self, individual, population):
        if not isinstance(self._rate, (list, tuple)):
            return self._rate

        fitness = [I.fitness for I in population.individuals]
        fit_max, fit_avg = np.max(fitness), np.mean(fitness)
        fit = individual.fitness
        # if fit_max < 0:
        #     return self._rate[1]
        if fit_max - fit_avg:
            return self._rate[1] if fit < fit_avg else self._rate[1] - (self._rate[1] - self._rate[0]) * (
                        fit_max - fit) / (fit_max - fit_avg)
        else:
            return (self._rate[0] + self._rate[1]) / 2.0

    def _mutate_positions(self, dimension):
        '''select num positions from dimension to mutate'''
        num = np.random.randint(dimension[0] * self.para_manager.SeqBitC + dimension[1] * self.para_manager.SeqBitP) + 1
        pos = np.random.choice(dimension[0] * self.para_manager.SeqBitC + dimension[1] * self.para_manager.SeqBitP,
                               num, replace=False)
        positions = np.zeros(dimension[0] * self.para_manager.SeqBitC +
                             dimension[1] * self.para_manager.SeqBitP).astype(np.bool_)
        positions[pos] = True
        seq_pos_i = np.where(pos < dimension[0] * self.para_manager.SeqBitC)
        seq_pos_p_i = np.where(pos >= dimension[0] * self.para_manager.SeqBitC)
        return positions, pos[seq_pos_i].copy(), pos[seq_pos_p_i].copy()

    def mutate(self, population, alpha=None):
        '''
        - population: population to be selected.
        - alpha: additional params
        '''
        self.mutation_num = 0
        # num = 0
        for individual in population.individuals:
            # num += 1
            rate = self._adaptive_rate(individual, population)
            if np.random.rand() > rate: continue
            self.mutation_num += 1
            pos = self._mutate_positions(individual.dimension)
            individual.solution = self.mutate_individual(individual, pos, alpha)
            individual.init_evaluation()  # reset evaluation
            individual.fitness = individual.evaluation
