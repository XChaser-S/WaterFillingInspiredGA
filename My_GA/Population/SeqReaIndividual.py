from GA.GAPopulation.Individual import Individual
from My_GA.util.HyperParameter import ParameterManager
import numpy as np
import copy


def decimal2gray(n, bits):
    # a = bin(n^(n>>1))[2:].zfill(bits)
    # if a == '100':
    #     print(n)
    return bin(n^(n>>1))[2:].zfill(bits)


def gray2decimal(gray_lst):
    bin_str = gray_lst[0]
    for i in range(len(gray_lst)-1):
        sum = int(gray_lst[0])
        for j in gray_lst[1:i+2]:
            sum = sum ^ int(j)
        bin_str = bin_str+str(sum)
    # if bin_str == '111':
    #     print(bin_str)
    return int(bin_str, 2)


class SeqReaIndividual(Individual):
    def __init__(self, seq_ranges, seq_rea_dimensions, para_manager: ParameterManager):
        self._dimension = seq_rea_dimensions
        self.num_bit = para_manager.SeqBit
        self.data_size = para_manager.DataSizes
        self.sub_band = para_manager.TotalBand/para_manager.NumChannel  # MHz
        self.noise = 10**(para_manager.Noise/10)  # mW/Hz
        self.csi = para_manager.CSI
        self.eta = para_manager.eta
        self.para_manager = para_manager
        self.assignment_channel_policy = {}
        self.evaluation = None
        self._solution = None
        # ToDo: add time and energy cost attr
        for i in range(para_manager.NumPL):
            self.assignment_channel_policy[f'PL{i+1}'] = []
        self.assignment_power_policy = copy.deepcopy(self.assignment_channel_policy)
        self.assignment_rate = copy.deepcopy(self.assignment_channel_policy)
        super().__init__(seq_ranges)
        self.fitness = copy.deepcopy(self.evaluation)

    def init_solution(self, ranges):
        self._ranges = ranges
        seq_range = self._ranges
        seq_dimension = self._dimension[0]
        real_dimension = self._dimension[1]
        self._solution = [np.zeros(seq_dimension*self.num_bit, np.str_), np.zeros(real_dimension)]
        self._solution[0] = self.init_sequence(seq_range, seq_dimension)
        self._solution[1] = self.init_real(real_dimension)

    def init_sequence(self, sequence_ranges, sequence_dimensions):
        decimal_solution = np.random.randint(0, sequence_ranges, sequence_dimensions)
        seq_solution = np.zeros(sequence_dimensions*self.num_bit, np.str_)
        for i in range(sequence_dimensions):
            # seq_solution[i*self.num_bit:(i+1)*self.num_bit] = list(bin(decimal_solution[i])[2:].zfill(self.num_bit))
            a = decimal2gray(decimal_solution[i], self.num_bit)
            seq_solution[i * self.num_bit:(i + 1) * self.num_bit] = list(a)
        return seq_solution

    @staticmethod
    def init_real(real_dimensions):
        real_solution = np.random.random(real_dimensions)

        return real_solution

    def solution2assignment(self):
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
                power = self._solution[1][channel] * self.para_manager.PowerMax[int(PL[2:])-1]
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

        if (False in (power_arr <= self.para_manager.PowerMax)) or (False in (rate_arr >= rate_thresh)):
            power_err = np.max([power_arr-self.para_manager.PowerMax, np.zeros(self.para_manager.NumPL)], axis=0)/\
                        self.para_manager.PowerMax
            rate_err = np.max([rate_thresh-rate_arr, np.zeros(self.para_manager.NumPL)], axis=0)/rate_thresh
            self.evaluation = -np.sum(power_err)-np.sum(rate_err)
        else:
            time_cost = np.mean(self.data_size*1024**3/rate_arr/self.para_manager.DTSyn)
            energy_cost = np.mean(power_arr*(self.data_size*1024**3/rate_arr)/(self.para_manager.PowerMax*self.para_manager.DTSyn))
            self.evaluation = 1/(self.eta*time_cost+(1-self.eta)*energy_cost)





