import time

import numpy as np

from MaxRate.MaxRateMain import MaxRateAssignment
from My_GA.MyMain import MyGA
from My_GA.util.HyperParameter import ParameterManager

from PowerIndividual import PowerIndividual
from PowerCrossover import PowerCrossover
from PowerMutation import PowerMutation
from PowerPopulation import PowerPopulation
from My_GA.Operators.MySelection import LinearRankingSelection


if __name__ == '__main__':
    para_manager = ParameterManager()
    channel_allocator = MaxRateAssignment(para_manager)
    channel_allocator.channel_assignment()
    I = PowerIndividual(para_manager.PowerLevels, para_manager.NumChannel,
                        para_manager, channel_allocator.assignment_channel_policy)
    P = PowerPopulation(I, para_manager.PopulationSize, para_manager.PreservationSize, para_manager)
    LRS = LinearRankingSelection()
    C = PowerCrossover([para_manager.AdaptiveMinQc, para_manager.AdaptiveMaxQc], para_manager)
    M = PowerMutation([para_manager.AdaptiveMinQm, para_manager.AdaptiveMaxQm], para_manager)
    g = MyGA(P, LRS, C, M)

    # build-in GA process
    s1 = time.time()
    # solve
    res = g.run(None, para_manager.MaxGeneration)
    # output
    s2 = time.time()

    power_arr = np.zeros(para_manager.NumPL)
    rate_arr = np.zeros(para_manager.NumPL)
    for PL in res.assignment_rate.keys():
        power_arr[int(PL[2:]) - 1] = np.sum(res.assignment_power_policy[PL])
        rate_arr[int(PL[2:]) - 1] = np.sum(res.assignment_rate[PL])

    g.data_logger['best_cost'].append(1 / res.evaluation)
    g.data_logger['best_PL_rate'].extend(rate_arr)

    energy_arr = power_arr * (para_manager.DataSizes * 1024 ** 3 / rate_arr)
    g.data_logger['best_PL_energy'].extend(energy_arr)

    # np.save(f'../experiment_data/UA-CGA/cost_{para_manager.NumPL}PL.npy', g.data_logger['best_cost'])
    # np.save(f'../experiment_data/UA-CGA/ave_elitism_fitness_{para_manager.NumPL}PL.npy',
    #         g.data_logger['average_elitism_fit'])
    # np.save(f'../experiment_data/UA-CGA/rate_{para_manager.NumPL}PL.npy', g.data_logger['best_PL_rate'])
    # np.save(f'../experiment_data/UA-CGA/energy_{para_manager.NumPL}PL.npy', g.data_logger['best_PL_energy'])
    print('GA solution: {0} in {1} seconds'.format(res.evaluation, s2 - s1))
