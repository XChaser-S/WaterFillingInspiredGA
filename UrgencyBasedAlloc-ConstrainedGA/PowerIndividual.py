from My_GA.Population.SeqReaIndividual import SeqIndividual
from My_GA.util.HyperParameter import ParameterManager
from My_GA.util.util import gray2decimal, decimal2gray
from util import power_adjustment, power_adjustment_without_sort
import numpy as np
import copy


class PowerIndividual(SeqIndividual):
    def __init__(self, seq_ranges, seq_rea_dimensions, para_manager: ParameterManager, channel_policy):
        super().__init__(seq_ranges, seq_rea_dimensions, para_manager)
        self.assignment_channel_policy = channel_policy
        self.init_evaluation()
        self.fitness = copy.deepcopy(self.evaluation)

    def init_solution(self, ranges):
        self._ranges = ranges
        seq_range = self._ranges
        seq_dimension = self._dimension
        self._solution = np.zeros(seq_dimension*self.power_bit, np.str_)
        self._solution = self.init_sequence(seq_range, seq_dimension)
        # power_adjustment(self)
        power_adjustment_without_sort(self)

    def init_sequence(self, sequence_ranges, sequence_dimensions):
        decimal_solution_power = np.random.randint(0, sequence_ranges, sequence_dimensions)
        seq_solution_power = np.zeros(sequence_dimensions * self.power_bit, np.str_)

        for i in range(sequence_dimensions):
            # seq_solution_channel[i*self.num_bit:(i+1)*self.num_bit] =
            # list(bin(decimal_solution_channel[i])[2:].zfill(self.num_bit))
            a = decimal2gray(decimal_solution_power[i], self.power_bit)
            seq_solution_power[i * self.power_bit:(i + 1) * self.power_bit] = list(a)

        return seq_solution_power

    def solution2assignment(self):
        for i in range(self.para_manager.NumPL):
            self.assignment_power_policy[f'PL{i+1}'] = []
        self.assignment_rate = copy.deepcopy(self.assignment_power_policy)
        # 遍历基因，获取子信道和功率分配情况，涉及到二进制解码和功率解码
        for PL, channels in self.assignment_channel_policy.items():
            for channel in channels:
                power_level = gray2decimal(self._solution[channel * self.power_bit:(channel + 1) * self.power_bit])
                power = self.power_space[int(PL[2:])-1, power_level]
                self.assignment_power_policy[PL].append(power)
                rate = self.sub_band*1e6 * \
                       np.log2(1+power*self.csi[channel][int(PL[2:])-1]/(self.noise*1e-3*self.sub_band*1e6))  # bit/s
                self.assignment_rate[PL].append(rate)

    def init_evaluation(self):
        if self.assignment_channel_policy == {}:
            pass
        else:
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





