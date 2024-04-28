import numpy as np


def generate_rayleigh_channel_gain(num_subcarriers, num_paths):
    # 生成每个多径分量的瑞利分布信道增益
    rayleigh_channel_gain = np.random.rayleigh(scale=1, size=(num_subcarriers, num_paths))
    return rayleigh_channel_gain


def apply_exponential_decay(rayleigh_channel_gain, decay_factor):
    # 根据指数衰减规律调整每个多径分量的幅度增益
    for l in range(rayleigh_channel_gain.shape[1]):
        rayleigh_channel_gain[:, l] = np.square(rayleigh_channel_gain[:, l]) * np.exp(-decay_factor * l)
    return rayleigh_channel_gain


def total_channel_gain(rayleigh_channel_gain):
    # 将所有多径分量的信道增益相加以获得总的信道增益
    return np.sum(rayleigh_channel_gain, axis=1)


class ParameterManager:
    def __init__(self):
        self.random_data_seed = 1629
        self.random_channel_seed = 1562

        self.PopulationSize = 300
        self.MaxGeneration = 800  # 600
        self.PreservationSize = 4

        self.AdaptiveMaxQc = 0.9
        self.AdaptiveMinQc = 0.7
        self.AdaptiveMaxQm = 0.1  # 0.1 0.3
        self.AdaptiveMinQm = 0.003

        self.NumPL = 8
        self.SeqBitC = self.get_SeqBit(self.NumPL)
        self.PowerMax = [2]*self.NumPL  # W
        self.PowerLevels = 10
        self.SeqBitP = self.get_SeqBit(self.PowerLevels)
        self.PowerSpace = self.get_PowerSpace()

        self.NumChannel = 64
        self.beta = 2  # 10*(3e8/(4*np.pi*50e6))**2
        self.distance = 100  # m
        self.TotalBand = 15  # MHz
        self.NumPath = 6
        self.PowerDecayFactor = 0.5
        self.CSI = self.get_CSI()

        self.DTSyn = 15  # s 15 20

        self.DataSizeMin = 0.5  # Gbit(1024)
        self.DataSizeMax = 0.9  # Gbit
        self.DataSizes = self.get_DataSizes()
        self.Noise = -200  # dBm/Hz

        self.eta = 0.8

    def get_SeqBit(self, n):
        bit_num = 0
        while 2**bit_num < n+1:
            bit_num += 1
        return bit_num

    def get_DataSizes(self):
        np.random.seed(self.random_data_seed)
        seeds = np.random.random(self.NumPL)
        data_min = np.array([self.DataSizeMin]*self.NumPL)
        data_max = np.array([self.DataSizeMax]*self.NumPL)
        data_size = data_min + (data_max - data_min) * seeds
        return data_size

    def get_CSI(self):
        csi_lst = []
        np.random.seed(self.random_channel_seed)
        for i in range(self.NumPL):
            channel_i = generate_rayleigh_channel_gain(self.NumChannel, self.NumPath)
            # channel_i = apply_exponential_decay(channel_i, self.PowerDecayFactor)
            # channel_i = total_channel_gain(self.beta * 1e-4 * channel_i**2)
            channel_i = apply_exponential_decay(channel_i, self.PowerDecayFactor)
            channel_i = total_channel_gain(self.beta * 1e-4 * channel_i)
            csi_lst.append(channel_i.reshape((len(channel_i), 1)))
        csi = np.concatenate(csi_lst, axis=1)
        return csi

    def get_PowerSpace(self):
        space = np.zeros((self.NumPL, self.PowerLevels+1))
        for i in range(self.NumPL):
            space[i, :-1] = self.PowerMax[i]*np.arange(0, 1, 1/self.PowerLevels)
        space[:, -1] = self.PowerMax
        return space
