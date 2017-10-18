import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk

from mpl_toolkits import mplot3d

class BatchFeeder:
    """ Simple iterator for feeding a subset of numpy matrix into tf network.
    validation data has same size of mini batch
     Parameter
    ----------------
    X: ndarray
    y: ndarray
    batch_size: mini batch size
    """

    def __init__(self, x_, batch_size, valid=False, ini_random=True):
        """check whether X and Y have the matching sample size."""
        self.n = len(x_)
        self.X = x_
        self.index = 0
        self.batch_size = batch_size
        self.base_index = np.arange(self.n)

    def next(self):
        if self.index + self.batch_size > self.n:
            self.index = 0
            self.base_index = self.randomize(self.base_index)
        ret_x = self.X[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return ret_x

    def randomize(self, index):
        np.random.shuffle(index)
        self.X = self.X[index]
        return index
    
def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    return voxels

def getAll(obj='airplane',train=True, is_local=False, cube_len=32, obj_ratio=1.0):
    objPath = "data/3DShapeNets/volumetric_data/" + obj + '/30/'
    objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    fileList = fileList[0:int(obj_ratio*len(fileList))]
    volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in fileList],dtype=np.bool)
    return volumeBatch

def plotVoxel(d, th = 0.5, size=(6,6)):
    temp = []
    bina = d > th
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                if bina[i, j, k]:
                    temp.append([i, j, k])
    temp = np.array(temp)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')

    n = 100
    colors = sns.color_palette("hls", 5)
    ax.scatter(temp[:,0], temp[:,1], temp[:,2], c=colors[0], marker=".", alpha=0.5, linewidth=0, s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
def interp(v1, v2, steps):
    vecs = []
    step = (v2-v1)/steps
    for i in range(steps):
        vecs.append(v1+step*i)
    vecs.append(v2)
    return vecs