import copy
import time
import pandas as pd
import numpy as np
from GA.GAProcess import GA
from My_GA.Population.SeqReaIndividual import SeqIndividual
from My_GA.Population.MyPopulation import MyPopulation
from My_GA.util.HyperParameter import ParameterManager
from My_GA.Operators.MySelection import LinearRankingSelection
from My_GA.Operators.MyMutation import MyMutation
from My_GA.Operators.MyCrossover import MyCrossover
from tqdm import tqdm


class MyGA(GA):
    def __init__(self, population, selection, crossover, mutation, fun_fitness=None):
        super().__init__(population, selection, crossover, mutation, fun_fitness)
        self.data_logger = {
            'average_elitism_fit': [],
            # 'best_total_throughput': [],
            'best_PL_energy': [],
            'best_cost': [],
            # 'data_size': [],
            'best_PL_rate': []
        }

    def run(self, fun_evaluation, gen=50):
        # initialize population
        self.population.initialize()
        self.population.preserve_elitism()

        # solving process
        for n in tqdm(range(1, gen + 1), position=0, leave=True):
            probe = {'cross_num': 0, 'mutation_num': 0, 'average_fit': 0, 'average_fit_elitism': 0}
            # selection
            self.population.individuals = self.selection.select(self.population)

            # crossover
            self.population.individuals = self.crossover.cross(self.population)
            probe['cross_num'] = [self.crossover.cross_num]
            # mutation
            self.mutation.mutate(self.population)
            probe['mutation_num'] = [self.mutation.mutation_num]
            # elitism mechanism:
            # evaluate and get the best individual in previous generation
            self.population.preserve_elitism()
            probe['average_fit_elitism'] = [np.mean([I.evaluation for I in self.population.preservation_individuals])]
            self.data_logger['average_elitism_fit'].extend(probe['average_fit_elitism'])
            probe['average_fit'] = [np.mean([I.evaluation for I in self.population.individuals])]
            print()
            print(pd.DataFrame(probe))
        evaluation = np.array([I.evaluation for I in self.population.preservation_individuals])
        # get the maximum position
        pos = np.argmax(evaluation)
        return copy.deepcopy(self.population.preservation_individuals[pos])


if __name__ == '__main__':
    para_manager = ParameterManager()
    I = SeqIndividual([para_manager.NumPL+1, para_manager.PowerLevels], [para_manager.NumChannel, para_manager.NumChannel], para_manager)
    P = MyPopulation(I, para_manager.PopulationSize, para_manager.PreservationSize, para_manager)
    LRS = LinearRankingSelection()
    C = MyCrossover([para_manager.AdaptiveMinQc, para_manager.AdaptiveMaxQc], para_manager)
    M = MyMutation([para_manager.AdaptiveMinQm, para_manager.AdaptiveMaxQm], para_manager)
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

    g.data_logger['best_cost'].append(1/res.evaluation)
    g.data_logger['best_PL_rate'].extend(rate_arr)

    energy_arr = power_arr*(para_manager.DataSizes*1024**3/rate_arr)
    g.data_logger['best_PL_energy'].extend(energy_arr)

    np.save(f'../experiment_data/Proposed/cost_{para_manager.NumPL}PL_Syn{para_manager.DTSyn}.npy', g.data_logger['best_cost'])
    np.save(f'../experiment_data/Proposed/ave_elitism_fitness_{para_manager.NumPL}PL_Syn{para_manager.DTSyn}.npy',
            g.data_logger['average_elitism_fit'])
    np.save(f'../experiment_data/Proposed/rate_{para_manager.NumPL}PL_Syn{para_manager.DTSyn}.npy', g.data_logger['best_PL_rate'])
    np.save(f'../experiment_data/Proposed/energy_{para_manager.NumPL}PL_Syn{para_manager.DTSyn}.npy', g.data_logger['best_PL_energy'])

    # np.save(f'../experiment_data/GA_NoSortedAdj/cost_{para_manager.NumPL}PL1.npy', g.data_logger['best_cost'])
    # np.save(f'../experiment_data/GA_NoSortedAdj/ave_elitism_fitness_{para_manager.NumPL}PL1.npy',
    #         g.data_logger['average_elitism_fit'])
    # np.save(f'../experiment_data/GA_NoSortedAdj/rate_{para_manager.NumPL}PL1.npy', g.data_logger['best_PL_rate'])
    # np.save(f'../experiment_data/GA_NoSortedAdj/energy_{para_manager.NumPL}PL1.npy', g.data_logger['best_PL_energy'])
    #
    print('GA solution: {0} in {1} seconds'.format(res.evaluation, s2 - s1))

# # plot
# cities.plot_cities(plt)
# cities.plot_path(plt, cities.solution)
# cities.plot_path(plt, res.solution)
# plt.show()
