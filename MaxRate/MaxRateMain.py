import copy

import numpy as np

from My_GA.util.HyperParameter import ParameterManager


class MaxRateAssignment:
    def __init__(self, para_manager: ParameterManager):
        self.pm = para_manager
        self.csi = copy.deepcopy(para_manager.CSI)
        self.csi_duplicate = copy.deepcopy(para_manager.CSI)
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
        self.time_cost_ave = None

    def channel_assignment(self):
        self.initialize_assignment()
        channel_left_n = self.pm.NumChannel-self.pm.NumPL
        while channel_left_n != 0:
            vip = self.get_vip()
            vip_csi = self.csi_duplicate[:, vip].copy()
            vip_channel_i = np.argsort(vip_csi)[-1]
            # vip_channel = vip_csi[vip_channel_i].copy()
            self.csi_duplicate[vip_channel_i, :] = 0
            channel_left_n -= 1
            self.assignment_channel_policy[f'PL{vip+1}'].append(vip_channel_i)
            vip_channel_n = len(self.assignment_channel_policy[f'PL{vip+1}'])
            vip_channels = self.csi[:, vip][self.assignment_channel_policy[f'PL{vip+1}']]
            self.assignment_power_policy[f'PL{vip+1}'] = [self.power_max[vip]/vip_channel_n]*vip_channel_n
            # self.assignment_power_policy[f'PL{vip + 1}'] = [1] * vip_channel_n
            self.assignment_rate[f'PL{vip+1}'] = list(self.sub_band * 1e6 * \
                                            np.log2(
                                                1 + np.array(self.assignment_power_policy[f'PL{vip+1}']) * vip_channels / (
                                                            self.noise * 1e-3 * self.sub_band * 1e6)))

    def random_channel_allocate(self):
        # channel_user = np.random.choice(self.pm.NumPL, 64)
        # for channel in range(len(channel_user)):
        #     self.assignment_channel_policy['PL'+str(channel_user[channel]+1)].append(channel)

        channel_user = np.random.choice(range(self.pm.NumPL+1), 64, replace=True)
        for channel in range(len(channel_user)):
            if channel_user[channel] != 0:
                self.assignment_channel_policy['PL' + str(channel_user[channel])].append(channel)
            else:
                print('zero occurred')
                continue



    def power_water_filling(self):
        for PL, channels in self.assignment_channel_policy.items():
            # pl_sorted_channels = np.sort(self.csi[:, int(PL[2:])-1][channels])
            # pl_sorted_channels_index = np.argsort(self.csi[:, int(PL[2:])-1][channels])
            pl_sorted_channels_index = np.array(channels)[::-1].copy()
            pl_sorted_channels = self.csi[:, int(PL[2:])-1][pl_sorted_channels_index]
            channel_n = len(pl_sorted_channels)
            p_k = 0
            for channel in pl_sorted_channels:
                p_c = (self.power_max[int(PL[2:])-1] +
                       np.sum(1/(pl_sorted_channels[-channel_n:]/(self.noise*1e3*self.sub_band))) -
                       channel_n/(channel/(self.noise*1e3*self.sub_band)))/channel_n
                if p_c <= 0:
                    channel_n -= 1
                    continue
                else:
                    self.assignment_channel_policy[PL] = pl_sorted_channels_index[-channel_n:]
                    p_k = p_c
                    break
            pl_sorted_channels = pl_sorted_channels[-channel_n:]
            pl_power_arr = np.zeros(channel_n)
            pl_power_arr[0] = p_k
            pl_power_arr[1:] = p_k + self.noise*self.sub_band*1e3*(1/pl_sorted_channels[0] - 1/pl_sorted_channels[1:])
            self.assignment_power_policy[PL] = [int(p/self.pm.PowerSpace[int(PL[2:])-1, 1]) * \
                                               self.pm.PowerSpace[int(PL[2:])-1, 1] for p in pl_power_arr]
            self.assignment_rate[PL] = list(self.sub_band*1e6 * \
                       np.log2(1+np.array(self.assignment_power_policy[PL])*pl_sorted_channels/(self.noise*1e-3*self.sub_band*1e6)))

    def equal_power_allocate(self, initialize=False):

        for PL, channels in self.assignment_channel_policy.items():
            if len(channels) >= self.pm.PowerLevels:
                channels_index = np.random.choice(range(len(channels)), self.pm.PowerLevels, replace=False)
                power_arr = np.zeros(len(channels))
                power_arr[channels_index] = np.array([self.power_max[int(PL[2:])-1]/self.pm.PowerLevels]*
                                                     self.pm.PowerLevels)
                power_lst = list(power_arr)
                # power_lst = [self.power_max[int(PL[2:])-1]/self.pm.PowerLevels]*self.pm.PowerLevels + [0.]*(
                #     len(channels)-self.pm.PowerLevels)
                self.assignment_power_policy[PL] = power_lst
            elif not initialize:
                power_lst = list(np.array(self.assignment_power_policy[PL]) /
                                 (self.power_max[int(PL[2:])-1]/self.pm.PowerLevels))
                power_lst = [int(power) * (self.power_max[int(PL[2:])-1]/self.pm.PowerLevels)
                             for power in power_lst]
                self.assignment_power_policy[PL] = power_lst
            else:
                allocated_channel_num = len(self.assignment_channel_policy[PL])
                power_lst = allocated_channel_num * [(self.power_max[int(PL[2:])-1]/self.pm.PowerLevels) *
                                                     int(self.pm.PowerLevels/allocated_channel_num)]
                self.assignment_power_policy[PL] = power_lst

            new_rate_lst = []
            # channel_state = []

            for i in range(len(channels)):
                h = self.csi[channels[i], int(PL[2:])-1]
                # channel_state.append(h)
                new_rate_lst.append(self.sub_band * 1e6 * np.log2(1 + self.assignment_power_policy[PL][i] *
                                                h / (self.noise * 1e-3 * self.sub_band * 1e6)))

            # new_rate_lst = list(self.sub_band * 1e6 * np.log2(1 + np.array(self.assignment_power_policy[PL]) *
            #                                     np.array(channel_state) / (self.noise * 1e-3 * self.sub_band * 1e6)))
            self.assignment_rate[PL] = new_rate_lst

    def evaluate(self):
        power_arr = self.get_power()
        rate_arr = self.get_rate()

        self.time_cost_ave = np.mean(self.pm.DataSizes * 1024 ** 3 / rate_arr)
        self.energy_cost_ave = np.mean(power_arr * (self.pm.DataSizes * 1024 ** 3 / rate_arr))
        self.integrated_cost = self.pm.eta * self.time_cost_ave / self.pm.DTSyn + \
                               (1 - self.pm.eta) * np.mean(power_arr * (self.pm.DataSizes * 1024 ** 3 / rate_arr)
                                                           / (np.array(self.pm.PowerMax) * self.pm.DTSyn))

        throughput = np.sum(rate_arr)*1e-9
        energy_sum = np.sum(power_arr * (self.pm.DataSizes * 1024 ** 3 / rate_arr))
        print(f'average_time:{self.time_cost_ave}s')
        print(f'average_energy:{self.energy_cost_ave}J')
        print(f'integrated_cost:{self.integrated_cost}')
        print(f'Throughput:{throughput}Gbps')
        print(f'energy_sum:{energy_sum}J')

        # np.save(f'../experiment_data/RA-EPA/cost_{para_manager.NumPL}PL.npy', self.integrated_cost)
        # np.save(f'../experiment_data/RA-EPA/rate_{para_manager.NumPL}PL.npy', throughput)
        # np.save(f'../experiment_data/RA-EPA/energy_{para_manager.NumPL}PL.npy', energy_sum)

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
            PL_i_csi = self.csi_duplicate[:, PL_i]
            PL_i_channel_j = np.argsort(PL_i_csi)[-1]
            PL_i_channel = PL_i_csi[PL_i_channel_j].copy()
            self.csi_duplicate[PL_i_channel_j, :] = 0
            self.assignment_channel_policy[f'PL{PL_i + 1}'].append(PL_i_channel_j)
            PL_i_channel_num = len(self.assignment_channel_policy[f'PL{PL_i + 1}'])
            power_i = self.power_max[PL_i] / PL_i_channel_num
            self.assignment_power_policy[f'PL{PL_i + 1}'] = [power_i] * PL_i_channel_num
            # power_i = 1
            # self.assignment_power_policy[f'PL{PL_i + 1}'] = [1] * PL_i_channel_num
            # 初始化速率
            rate_i = self.sub_band*1e6 * \
                       np.log2(1+power_i*PL_i_channel/(self.noise*1e-3*self.sub_band*1e6))
            self.assignment_rate[f'PL{PL_i + 1}'].append(rate_i)


if __name__ == '__main__':
    para_manager = ParameterManager()
    agent = MaxRateAssignment(para_manager)
    # agent.channel_assignment()
    # agent.equal_power_allocate()
    agent.random_channel_allocate()
    agent.equal_power_allocate(initialize=True)
    # agent.power_water_filling()
    agent.evaluate()
