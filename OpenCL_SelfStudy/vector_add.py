import pyopencl as cl
import numpy

from time import time

TOL = 0.001

SIZE = (1024, 1024)

kernelsource = 
__kernel void vaad(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsinged int count)
{
    int id = get_global_id(0);
    if (i < count){
        c[i] = a[i] + b[i]
    }
}
