# RSMT utils
'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
import platform
import heapq
class ArrayType:
    def __init__(self, type):
        self.type = type

    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError('Can\'t convert % s' % typename)

    # Cast from lists / tuples
    def from_list(self, param):
        val = ((self.type) * len(param))(*param)
        return val

    from_tuple = from_list
    from_array = from_list
    from_ndarray = from_list
class Evaluator:
    def __init__(self):
        cdll_names = {
            'Linux': 'Getfitness.so',
            'Windows': 'C:\\Users\\LOA\\Desktop\\VideoUnderstanding\\XSMT\\Solving-VLSI-DRL\\Getfitness.dll'
        }
        #path = os.getcwd() + '/al/libeval.so'
        eval_mod = ctypes.cdll.LoadLibrary(cdll_names[platform.system()])
        DoubleArray = ArrayType(ctypes.c_double)
        IntArray = ArrayType(ctypes.c_int)

      #  eval_mod.gst_open_geosteiner()

        eval_func = eval_mod.Getfitness
        eval_func.argtypes = (DoubleArray, IntArray, ctypes.c_int)
        eval_func.restype = ctypes.c_double
        self.eval_func = eval_func
    def eval_batch(self, input_batch, output_batch):
        lengths = []
        batch_size = len(input_batch)
        for i in range(batch_size):
            degree = np.size(input_batch,1)
            V=np.concatenate((np.zeros(2),input_batch[i].reshape(-1)))
            xes = np.concatenate((np.zeros(1,dtype=np.int64),output_batch[i].reshape(-1)))
            lengths.append(self.eval_func(V, xes, degree))
        return np.array(lengths)
def transform_inputs(inputs, t):
    # 0 <= t <= 7
    xs = inputs[:, :, 0]
    ys = inputs[:, :, 1]
    if t >= 4:
        temp = xs
        xs = ys
        ys = temp
    if t % 2 == 1:
        xs = 1 - xs
    if t % 4 >= 2:
        ys = 1 - ys
    return np.stack([xs, ys], -1)