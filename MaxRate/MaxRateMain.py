import copy

import numpy as np

from My_GA.util.HyperParameter import ParameterManager

class MaxRateAssignment:
    def __init__(self, para_manager: ParameterManager):
        self.pm = para_manager
        self.csi = copy.deepcopy(para_manager.CSI)
        self.sub_band = para_manager.TotalBand/para_manager.NumChannel
        self.noise = 10**(para_manager.Noise/10)  # mW/Hz
        self.assignment_channel_policy = {}
        for i in range(para_manager.NumPL):
            self.assignment_channel_policy[f'PL{i + 1}'] = []
        self.assignment_power_policy = copy.deepcopy(self.assignment_channel_policy)
        self.assignment_rate = copy.deepcopy(self.assignment_channel_policy)

        self.rate_thresh = self.pm.DataSizes * 1024 ** 3 / self.pm.DTSyn
        self.power_max = np.array(self.pm.PowerMax)

        self.integrated_cost = None
        self.energy_cost = None
        self.time_cost = None

    def channel_assignment(self):
        self.initialize_assignment()
        channel_left_n = self.pm.NumChannel-self.pm.NumPL
        while channel_left_n != 0:
            vip = self.get_vip()
            vip_csi = self.csi[:, vip]
            vip_channel_i = np.argsort(vip_csi)[-1]
            vip_channel = vip_csi[vip_channel_i].copy()
            self.csi[vip_channel_i, :] = 0
            channel_left_n -= 1
            self.assignment_channel_policy[f'PL{vip+1}'].append(vip_channel)
            vip_channel_n = len(self.assignment_channel_policy[f'PL{vip+1}'])
            self.assignment_power_policy[f'PL{vip+1}'] = [self.power_max[vip]/vip_channel_n]*vip_channel_n
        power_arr = self.get_power()
        rate_arr = self.get_rate()
        self.time_cost = np.mean(self.pm.DataSizes * 1024 ** 3 / rate_arr)
        self.energy_cost = np.mean(power_arr * (self.pm.DataSizes * 1024 ** 3 / rate_arr))
        self.integrated_cost = self.pm.eta*self.time_cost / self.pm.DTSyn +\
                               (1-self.pm.eta)*np.mean(power_arr * (self.pm.DataSizes * 1024 ** 3 / rate_arr)
                                                       /(np.array(self.pm.PowerMax) * self.pm.DTSyn))
        print(f'average_time:{self.time_cost}s')
        print(f'average_energy:{self.energy_cost}J')
        print(f'integrated_cost:{self.integrated_cost}')

    def get_rate(self):
        rate_arr = np.zeros(self.pm.NumPL)
        for PL, rate_lst in self.assignment_rate.items():
            rate_arr[int(PL[2:])-1] = np.sum(rate_lst)
        return rate_arr

    def get_power(self):
        power_arr = np.zeros(self.pm.NumPL)
        for PL, power_lst in self.assignment_power_policy.items():
            power_arr[int(PL[2:]) - 1] = np.sum(power_lst)
        return power_arr

    def get_vip(self):
        rate = self.get_rate()
        vip = np.argsort(rate/self.rate_thresh)[0]
        return vip

    def initialize_assignment(self):
        for PL_i in range(self.pm.NumPL):
            # 为每一个用户选择最优信道
            PL_i_csi = self.csi[:, PL_i]
            PL_i_channel_j = np.argsort(PL_i_csi)[-1]
            PL_i_channel = PL_i_csi[PL_i_channel_j].copy()
            self.csi[PL_i_channel_j, :] = 0
            self.assignment_channel_policy[f'PL{PL_i + 1}'].append(PL_i_channel)
            # 等功率分配
            PL_i_channel_num = len(self.assignment_channel_policy[f'PL{PL_i + 1}'])
            power_i = self.power_max[PL_i] / PL_i_channel_num
            self.assignment_power_policy[f'PL{PL_i + 1}'] = [power_i] * PL_i_channel_num
            # 初始化速率
            rate_i = self.sub_band*1e6 * \
                       np.log2(1+power_i*PL_i_channel/(self.noise*1e-3*self.sub_band*1e6))
            self.assignment_rate[f'PL{PL_i + 1}'].append(rate_i)


if __name__ == '__main__':
    para_manager = ParameterManager()
    agent = MaxRateAssignment(para_manager)
    agent.channel_assignment()
