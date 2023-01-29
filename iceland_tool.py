import pandas as pd
import numpy as np
import pickle
import wfdb

import os
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import RobustScaler
from scipy.signal import butter, lfilter, medfilt
from config import Config
import data_util as du



    
def get_names():#contain_abnormal = True
    file_dir = Config.ICELAND_DIR +'/'
    record_file = file_dir + 'RECORDS'
    fp = open(record_file)
    ID = fp.read()
    ID = np.asanyarray(ID.split('\n'))[:-1]
    fp.close()
    return ID


def get_signal(name = None,channels = None,filter = None,**options):
    
    data_dir = Config.ICELAND_DIR +'/'
    #先滤波后再降采样
    downsample = options.pop('down_sample',True)
    target_fs=  options.pop('target_fs', 20)
    normalize = options.pop('normalize', False)

    sample_freq = 200    

    
    signal = du.get_signal(data_dir+name, [get_correct_channel(channels[0])])
    if normalize:
        signal = du.standardization(signal)
    if len(channels) == 2:
        signal_2 = du.get_signal(data_dir+name,[get_correct_channel(channels[1])])
        if normalize:
            signal_2 = du.standardization(signal_2)
        signal = signal_2 - signal
        
        
    if filter is not None:
        signal = du.butter_bandpass(signal.flatten(), filter[0], filter[1], fs = sample_freq, order = 4, butterworth = True) #滤波

    if downsample:
        signal = du.re_sample(signal, sample_freq, target_fs)

    return signal.flatten()

    
def get_correct_channel(channel):
    channels = [1,10,11,12,13,14,15,16,2,3,4,5,6,7,8,9]
    for index in range(len(channels)):
        if channels[index] == channel:
            return index
    return        



      
def islabor(name):
    temp = name[7] #表明怀孕或临产，p为怀孕期间
    if temp == 'p':
        labor = 0
    else:
        labor = 1
    return labor



        
