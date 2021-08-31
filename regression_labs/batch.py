import numpy as np
from sample_generator import SampleGenerator
from typing import Tuple


class Batch:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.batch_size = 50
        self.g = SampleGenerator()

    def reset_batch(self) -> None:
        self.x_data = []
        self.y_data = []

    def add_non_linear_sample(self) -> Tuple[float, float]:
        x = np.random.random()
        y = self.g.generate_non_linear_samples(x)
        self.x_data.append(x)
        self.y_data.append(y)
        return x, y

    def make_nonlinear_batch_data(self) -> None:
        """ 
        Generate a batch of non linear data and store it into numpy structures
        """
        self.reset_batch()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = self.g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def make_linear_batch_data(self) -> None:
        """ 
        Generate a batch of linear data and store it into numpy structures
        """
        self.reset_batch()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = self.g.generate_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)
