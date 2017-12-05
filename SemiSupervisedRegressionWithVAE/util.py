import numpy as np

class BatchFeeder:
    """ Simple iterator for feeding a subset of numpy matrix into tf network.
    validation data has same size of mini batch
     Parameter
    ----------------
    X: ndarray
    y: ndarray
    batch_size: mini batch size
    """

    def __init__(self, x_, y_, batch_size):
        """check whether X and Y have the matching sample size."""
        assert x_.shape[0] == y_.shape[0]
        self.n = x_.shape[0]
        self.X = x_
        self.y = y_
        self.index = 0
        self.batch_size = batch_size
        self.batch_num = int(np.ceil(self.n*1.0/self.batch_size))
    
    def next(self):
        if self.index + self.batch_size > self.n:
            self.index = 0
            self.randomize()
        ret_x = self.X[self.index:self.index+self.batch_size]
        ret_y = self.y[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return ret_x, ret_y
    
    def randomize(self):
        indexes = np.arange(self.n)
        np.random.shuffle(indexes)
        self.y = self.y[indexes]
        self.X = self.X[indexes]