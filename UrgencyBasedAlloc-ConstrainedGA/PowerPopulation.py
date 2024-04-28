from My_GA.Population.MyPopulation import MyPopulation
import numpy as np


class PowerPopulation(MyPopulation):

    def initialize(self):
        IndvClass = self.individual.__class__
        self.individuals = np.array(
            [IndvClass(self.individual.ranges, self.individual.dimension,
                       self.para_manager, self.individual.assignment_channel_policy)
             for i in range(self.size)], dtype=IndvClass)

