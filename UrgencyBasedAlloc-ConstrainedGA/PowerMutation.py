import copy
from My_GA.Operators.MyMutation import MyMutation
from PowerIndividual import PowerIndividual
from My_GA.util.HyperParameter import ParameterManager
from My_GA.util.util import gray2decimal, decimal2gray
from util import power_adjustment, power_adjustment_without_sort
import numpy as np


class PowerMutation(MyMutation):
    def __init__(self, rate, para_manager: ParameterManager):
        super().__init__(rate, para_manager)
        self._individual_class = [PowerIndividual]

    @staticmethod
    def mutate_individual(individual, positions, alpha):
        '''
        get mutated solution based on the selected individual:
            - individual: the selected individual
            - positions : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: the mutated solution
        '''
        # 设计掩膜
        solution = individual.solution.copy()
        seq_pos_p_mask = positions[0][-individual.dimension * individual.power_bit:]
        seq_pos_p_index = positions[1]

        # 取反
        solution[seq_pos_p_mask] = ((~solution[seq_pos_p_mask].astype(np.int32)) + 2).astype(np.str_)
        # fix the illegal mutation
        for pos in (seq_pos_p_index / individual.power_bit).astype(np.int32):
            power_level = gray2decimal(solution[pos*individual.power_bit:(pos+1)*individual.power_bit])
            if power_level > individual.para_manager.PowerLevels:
                # solution[0][pos * individual.num_bit:(pos + 1) * individual.num_bit] = \
                #     list(bin(individual.para_manager.NumPL)[2:].zfill(individual.num_bit))
                # if decimal2gray(individual.para_manager.NumPL, individual.num_bit) == '111':
                #     print(pos)
                power_level = np.random.choice(individual.para_manager.PowerLevels)+1
                solution[pos * individual.power_bit:(pos + 1) * individual.power_bit] = \
                    list(decimal2gray(power_level, individual.power_bit))
        # power_adjustment(individual)
        # power_adjustment_without_sort(individual)

        return copy.deepcopy(solution)

    def _mutate_positions(self, dimension):
        '''select num positions from dimension to mutate'''
        num = np.random.randint(dimension * self.para_manager.SeqBitP) + 1
        pos = np.random.choice(dimension * self.para_manager.SeqBitP,
                               num, replace=False)
        positions = np.zeros(dimension * self.para_manager.SeqBitP).astype(np.bool_)
        positions[pos] = True
        return positions, pos.copy()

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
