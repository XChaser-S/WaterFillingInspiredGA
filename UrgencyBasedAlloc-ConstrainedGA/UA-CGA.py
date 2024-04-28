import time

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

    print('GA solution: {0} in {1} seconds'.format(res.evaluation, s2 - s1))
