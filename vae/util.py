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

    def __init__(self, x_, y_, batch_size, valid=False, ini_random=True):
        """check whether X and Y have the matching sample size."""
        assert len(x_) == len(y_)
        self.n = len(x_)
        self.X = x_
        self.y = y_
        self.index = 0
        # self.base_index = np.arange(len(X))
        if ini_random:
            _ = self.randomize(np.arange(len(x_)))
        if valid:
            self.create_validation(batch_size)
        self.batch_size = batch_size
        self.base_index = np.arange(self.n)
        self.val = None

    def create_validation(self, batch_size):
        self.val = (self.X[-1*int(batch_size):], self.y[-1*int(batch_size):])
        self.X = self.X[:-1*int(batch_size)]
        self.y = self.y[:-1*int(batch_size)]
        self.n = len(self.X)-int(batch_size)

    def next(self):
        if self.index + self.batch_size > self.n:
            self.index = 0
            self.base_index = self.randomize(self.base_index)
        ret_x = self.X[self.index:self.index+self.batch_size]
        ret_y = self.y[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return ret_x, ret_y

    def randomize(self, index):
        np.random.shuffle(index)
        self.y = self.y[index]
        self.X = self.X[index]
        return index