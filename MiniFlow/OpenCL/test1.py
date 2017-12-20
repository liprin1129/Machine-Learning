import pyopencl as cl
from pyopencl import array
import numpy as np
from time import time


if __name__ == "__main__":

    vector = np.random.rand(128, 1)

    matrix = np.random.rand(784, 128)
 2 12 
    rtime = time()     
    output = np.dot(matrix, vector)

    #print output

    rtime = time() - rtime                                                                                             
    print "The kernel ran in", rtime, "seconds"    
