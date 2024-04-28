import numpy as np

from My_GA.util.util import decimal2gray


def power_adjustment(individual):
    power_max = individual.para_manager.PowerMax
    individual.solution2assignment()
    channel_assignment = individual.assignment_channel_policy
    power_assignment = individual.assignment_power_policy
    csi = individual.csi
    for PL, power_lst in power_assignment.items():
        if not power_lst == []:
            power_sum = np.sum(power_lst)
            power_lst = np.array(power_lst.copy())
            power_lst_sorted = np.sort(power_lst)
            PL_channel_sorted_i = np.argsort(csi[:, int(PL[2:]) - 1][np.array(channel_assignment[PL])])
            power_lst[PL_channel_sorted_i] = power_lst_sorted

            if power_sum > power_max[int(PL[2:])-1]:
                for channel_i in PL_channel_sorted_i:
                    if np.sum(power_lst) > power_max[int(PL[2:])-1]:
                        # 调整功率
                        adjusted_power = np.max([0, power_lst[channel_i]+power_max[int(PL[2:])-1]-np.sum(power_lst)])
                        power_lst[channel_i] = adjusted_power
                        # 调整solution功率水平
                        power_level = round(adjusted_power/(power_max[int(PL[2:])-1]/individual.para_manager.PowerLevels))
                        channel = channel_assignment[PL][channel_i]
                        adjusted_power_gray = decimal2gray(power_level, individual.power_bit)
                        individual.solution[channel*individual.power_bit:(channel+1)*individual.power_bit] = \
                            list(adjusted_power_gray)
                    else:
                        power_level = round(
                            power_lst[channel_i] / (power_max[int(PL[2:]) - 1] / individual.para_manager.PowerLevels))
                        channel = channel_assignment[PL][channel_i]
                        power_gray = decimal2gray(power_level, individual.power_bit)
                        individual.solution[channel * individual.power_bit:(channel + 1) * individual.power_bit] = \
                            list(power_gray)
            else:
                for channel_i in PL_channel_sorted_i:
                    power_level = round(
                        power_lst[channel_i] / (power_max[int(PL[2:]) - 1] / individual.para_manager.PowerLevels))
                    channel = channel_assignment[PL][channel_i]
                    power_gray = decimal2gray(power_level, individual.power_bit)
                    individual.solution[channel * individual.power_bit:(channel + 1) * individual.power_bit] = \
                        list(power_gray)


def power_adjustment_without_sort(individual):
    power_max = individual.para_manager.PowerMax
    individual.solution2assignment()
    channel_assignment = individual.assignment_channel_policy
    power_assignment = individual.assignment_power_policy
    csi = individual.csi
    # for PL, power_lst in power_assignment.items():
    #     if not power_lst == []:
    #         power_sum = np.sum(power_lst)
    #         power_lst = np.array(power_lst.copy())
    #         PL_channel_sorted_i = np.argsort(csi[:, int(PL[2:]) - 1][np.array(channel_assignment[PL])])
    #
    #         if power_sum > power_max[int(PL[2:]) - 1]:
    #             for channel_i in PL_channel_sorted_i:
    #                 if np.sum(power_lst) > power_max[int(PL[2:]) - 1]:
    #                     # 调整功率
    #                     adjusted_power = np.max(
    #                         [0, power_lst[channel_i] + power_max[int(PL[2:]) - 1] - np.sum(power_lst)])
    #                     power_lst[channel_i] = adjusted_power
    #                     # 调整solution功率水平
    #                     power_level = round(
    #                         adjusted_power / (power_max[int(PL[2:]) - 1] / individual.para_manager.PowerLevels))
    #                     channel = channel_assignment[PL][channel_i]
    #                     adjusted_power_gray = decimal2gray(power_level, individual.power_bit)
    #                     individual.solution[channel * individual.power_bit:(channel + 1) * individual.power_bit] = \
    #                         list(adjusted_power_gray)
    for PL, power_lst in power_assignment.items():
        if not power_lst == []:
            power_lst = np.array(power_lst.copy())

            while np.sum(power_lst) > power_max[int(PL[2:]) - 1]:
                channel_i = np.random.choice(range(len(channel_assignment[PL])), replace=False)
                # 调整功率
                adjusted_power = np.max(
                    [0, power_lst[channel_i] + power_max[int(PL[2:]) - 1] - np.sum(power_lst)])
                power_lst[channel_i] = adjusted_power
                # 调整solution功率水平
                power_level = round(
                    adjusted_power / (power_max[int(PL[2:]) - 1] / individual.para_manager.PowerLevels))
                channel = channel_assignment[PL][channel_i]
                adjusted_power_gray = decimal2gray(power_level, individual.power_bit)
                individual.solution[channel * individual.power_bit:(channel + 1) * individual.power_bit] = \
                    list(adjusted_power_gray)