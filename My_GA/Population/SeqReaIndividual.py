from GA.GAPopulation.Individual import Individual
from My_GA.util.HyperParameter import ParameterManager
from My_GA.util.util import power_adjustment, gray2decimal, decimal2gray
import numpy as np
import copy


class SeqIndividual(Individual):
    def __init__(self, seq_ranges, seq_rea_dimensions, para_manager: ParameterManager):
        self._dimension = seq_rea_dimensions
        self.num_bit = para_manager.SeqBitC
        self.power_bit = para_manager.SeqBitP
        self.data_size = para_manager.DataSizes
        self.sub_band = para_manager.TotalBand/para_manager.NumChannel  # MHz
        self.noise = 10**(para_manager.Noise/10)  # mW/Hz
        self.csi = para_manager.CSI
        self.power_space = para_manager.PowerSpace
        self.eta = para_manager.eta
        self.para_manager = para_manager
        self.assignment_channel_policy = {}
        self.evaluation = None
        self._solution = None
        # ToDo: add time and energy cost attr
        self.energy = None
        self.time = None
        self.assignment_power_policy = {}
        self.assignment_rate = {}
        super().__init__(seq_ranges)
        self.fitness = copy.deepcopy(self.evaluation)

    def init_solution(self, ranges):
        self._ranges = ranges
        seq_range = self._ranges
        seq_dimension = self._dimension
        self._solution = [np.zeros(seq_dimension[0]*self.num_bit, np.str_), np.zeros(seq_dimension[1]*self.power_bit, np.str_)]
        self._solution = self.init_sequence(seq_range, seq_dimension)
        power_adjustment(self)

    def init_sequence(self, sequence_ranges, sequence_dimensions):
        # decimal_solution_channel = np.random.randint(1, sequence_ranges[0], sequence_dimensions[0])
        decimal_solution_channel = np.random.randint(0, sequence_ranges[0], sequence_dimensions[0])
        decimal_solution_power = np.random.randint(0, sequence_ranges[1], sequence_dimensions[1])
        seq_solution_channel = np.zeros(sequence_dimensions[0] * self.num_bit, np.str_)
        seq_solution_power = np.zeros(sequence_dimensions[1] * self.power_bit, np.str_)
        for i in range(sequence_dimensions[0]):
            # seq_solution_channel[i*self.num_bit:(i+1)*self.num_bit] =
            # list(bin(decimal_solution_channel[i])[2:].zfill(self.num_bit))
            a = decimal2gray(decimal_solution_channel[i], self.num_bit)
            seq_solution_channel[i * self.num_bit:(i + 1) * self.num_bit] = list(a)

        for i in range(sequence_dimensions[1]):
            # seq_solution_channel[i*self.num_bit:(i+1)*self.num_bit] =
            # list(bin(decimal_solution_channel[i])[2:].zfill(self.num_bit))
            a = decimal2gray(decimal_solution_power[i], self.power_bit)
            seq_solution_power[i * self.power_bit:(i + 1) * self.power_bit] = list(a)

        return [seq_solution_channel, seq_solution_power]

    @staticmethod
    def init_real(real_dimensions):
        real_solution = np.random.random(real_dimensions)

        return real_solution

    def solution2assignment(self):
        for i in range(self.para_manager.NumPL):
            self.assignment_channel_policy[f'PL{i+1}'] = []
        self.assignment_power_policy = copy.deepcopy(self.assignment_channel_policy)
        self.assignment_rate = copy.deepcopy(self.assignment_channel_policy)
        # 遍历基因，获取子信道和功率分配情况，涉及到二进制解码和功率解码
        for channel_i in range(self._dimension[0]):
            # PL_index = ''
            # for i in self._solution[0][channel_i * self.num_bit:(channel_i + 1) * self.num_bit]:
            #     PL_index += i
            # PL_index = int(PL_index, 2)
            # if list(self._solution[0][channel_i * self.num_bit:(channel_i + 1) * self.num_bit]) == list('100'):
            #     print()
            PL_index = gray2decimal(self._solution[0][channel_i * self.num_bit:(channel_i + 1) * self.num_bit])
            if PL_index == 0:
                continue
            else:
                self.assignment_channel_policy[f'PL{PL_index}'].append(channel_i)

        for PL, channels in self.assignment_channel_policy.items():
            for channel in channels:
                power_level = gray2decimal(self._solution[1][channel * self.power_bit:(channel + 1) * self.power_bit])
                power = self.power_space[int(PL[2:])-1, power_level]
                self.assignment_power_policy[PL].append(power)
                rate = self.sub_band*1e6 * \
                       np.log2(1+power*self.csi[channel][int(PL[2:])-1]/(self.noise*1e-3*self.sub_band*1e6))  # bit/s
                self.assignment_rate[PL].append(rate)

    def init_evaluation(self):
        self.solution2assignment()
        power_arr = np.zeros(self.para_manager.NumPL)
        rate_arr = np.zeros(self.para_manager.NumPL)
        for PL in self.assignment_rate.keys():
            power_arr[int(PL[2:])-1] = np.sum(self.assignment_power_policy[PL])
            rate_arr[int(PL[2:])-1] = np.sum(self.assignment_rate[PL])
        rate_thresh = self.data_size*1024**3/self.para_manager.DTSyn

        if (False in (power_arr <= np.array(self.para_manager.PowerMax)+0.001)) or (False in (rate_arr >= rate_thresh)):
            power_err = np.max([power_arr-self.para_manager.PowerMax, np.zeros(self.para_manager.NumPL)], axis=0)/\
                        self.para_manager.PowerMax
            rate_err = np.max([rate_thresh-rate_arr, np.zeros(self.para_manager.NumPL)], axis=0)/rate_thresh
            self.evaluation = -np.sum(power_err)-np.sum(rate_err)
        else:
            time_cost = np.mean(self.data_size*1024**3/rate_arr/self.para_manager.DTSyn)
            energy_cost = np.mean(power_arr*(self.data_size*1024**3/rate_arr)/(np.array(self.para_manager.PowerMax)*self.para_manager.DTSyn))
            self.evaluation = 1/(self.eta*time_cost+(1-self.eta)*energy_cost)





