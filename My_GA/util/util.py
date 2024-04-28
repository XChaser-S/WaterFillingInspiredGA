import numpy as np


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
                        individual.solution[1][channel*individual.power_bit:(channel+1)*individual.power_bit] = \
                            list(adjusted_power_gray)
                    else:
                        power_level = round(
                            power_lst[channel_i] / (power_max[int(PL[2:]) - 1] / individual.para_manager.PowerLevels))
                        channel = channel_assignment[PL][channel_i]
                        power_gray = decimal2gray(power_level, individual.power_bit)
                        individual.solution[1][channel * individual.power_bit:(channel + 1) * individual.power_bit] = \
                            list(power_gray)
            else:
                for channel_i in PL_channel_sorted_i:
                    power_level = round(
                        power_lst[channel_i] / (power_max[int(PL[2:]) - 1] / individual.para_manager.PowerLevels))
                    channel = channel_assignment[PL][channel_i]
                    power_gray = decimal2gray(power_level, individual.power_bit)
                    individual.solution[1][channel * individual.power_bit:(channel + 1) * individual.power_bit] = \
                        list(power_gray)
