import data_util as du
from config import Config
import numpy as np
from iceland.contraction_time import contraction_time
import matplotlib.pyplot as plt


def plot_signal(signal, record_time, fs = 6, show = False):
    x = np.linspace(0 , signal.shape[0], num = signal.shape[0])#/(fs*60)
    x_label = record_time.get_x_label(signal.shape[0], fs)
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize = (20.0,4))#80,50
    plt.subplots_adjust(left=0.13, bottom=0.14, right=0.96, top=0.85,
                wspace=None, hspace=None)
    plt.plot(x,signal,linewidth=2)
    
    x_ = x[fs*(60 - record_time.second):]
    x_label_ = x_label[fs*(60 - record_time.second):]
    plt.xticks(x_[::fs*60],x_label_[::fs*60])#将x轴改为文字
    if show:
        plt.show()

# def get_intervals(signal,fs, thr = 30):
#     minute_interval = []
#     minutes = du.get_total_minute(len(signal), fs)
#     minutes = int(minutes-1)
#     split = 1
#     while minutes/split > thr:
#         split+=1
#     for i in range(split):
#         if i == 0:
#             onset =1
#         else:
#             onset = minutes//split*i
#         if i == split-1:
#             offset = minutes
#         else:
#             offset = minutes//split*(i+1)

#         minute_interval.append([onset, offset])
#     return minute_interval

def get_intervals(signal,fs, thr = 30):
    minute_interval = []
    minutes = du.get_total_minute(len(signal), fs)
    #minutes = int(minutes-1) 
    split = 1
    while minutes/split > thr:
        split+=1
    for i in range(split):
        if i == 0:
            onset =0
        else:
            onset = minutes/split*i
        if i == split-1:
            offset = minutes
        else:
            offset = minutes/split*(i+1)
        
        minute_interval.append([onset, offset])
    return minute_interval

# def plot_signal(signal, record_time, fs = 250, show = False):
#     x = np.linspace(0 , signal.shape[0]/(fs*60), num = signal.shape[0])
#     x_label = record_time.get_x_label(signal.shape[0], fs)
#     plt.figure()
#     fig, ax = plt.subplots(1, 1, figsize = (20.0,4))#80,50
#     plt.subplots_adjust(left=0.13, bottom=0.14, right=0.96, top=0.85,
#                 wspace=None, hspace=None)
#     plt.plot(x,signal,linewidth=2)
#     x_ = x[fs*(60 - record_time.second):]
#     x_label_ = x_label[fs*(60 - record_time.second):]
#     plt.xticks(x_[::fs*60],x_label_[::fs*60])#将x轴改为文字
#     if show:
#         plt.show()
    
# def get_names(abnormal = False):
#     file_dir = get_dir()
#     record_file = file_dir + 'RECORDS'
#     fp = open(record_file)
#     ID = fp.read()
#     ID = np.asanyarray(ID.split('\n'))[:-1]
#     abnormal_names = get_abnormal_names(abnormal)
#     for name in abnormal_names:
#         index = np.argwhere(ID == name)
#         if len(index) == 0:
#             print('error:', name)
#         else:
#             ID = np.delete(ID,  index[0][0] )
#     fp.close()
#     return ID

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
 
# def Fun(p,x):                        # 定义拟合函数形式
#     a1,a2,a3 = p
#     return a1*x**2+a2*x+a3
# def error (p,x,y):                    # 拟合残差
#     return Fun(p,x)-y 
# def get_y_fit(x, y):  #get_coefficient
#     p_value = [-2,5,10] # 原始数据的参数
#     p0 = [0.1,-0.01,100] # 拟合的初始参数设置
#     para =leastsq(error, p0, args=(x,y)) # 进行拟合
#     y_fitted = Fun (para[0],x) # 画出拟合后的曲线
    
#     return y_fitted, para[0]

 
def Fun(p,x):                        # 定义拟合函数形式
    a1,a2= p
    return a1*x+a2
def error(p,x,y):                    # 拟合残差
    return Fun(p,x)-y

def get_y_fit(x, y):  #get_coefficient
    p_value = [-2,10] # 原始数据的参数
    p0 = [0.1,10] # 拟合的初始参数设置
    para =leastsq(error, p0, args=(x,y)) # 进行拟合
    y_fitted = Fun (para[0],x) # 画出拟合后的曲线
    
    return y_fitted, para[0]


import numpy as np
from scipy.stats import pearsonr
import random

def calc_cross_correlation(X,Y):
    corr,_ = pearsonr(X,Y)
    return corr

# def calc_cross_correlation(X,Y):
#     #计算相关系数，未使用
#     N = len(X)
#     R_x = 1/N*np.sum(X**2)
#     R_y = 1/N*np.sum(Y**2)
#     R_xy = 1/N*np.sum(np.multiply(X,Y))
#     import math
#     corr = R_xy / math.sqrt(R_x * R_y) 
    
#     return corr
def rmse(X,Y):
    return np.sqrt(np.sum(np.power(Y-X, 2))/len(X))

def evaluate_corr(X,Y):
    corr = calc_cross_correlation(X,Y)
    err = rmse(X,Y)
    print('corr:', np.round(corr,3), 'rmse:',np.round(err,3))
    

        
        
        
def plot_mark(signal,start_x, end_x):
#     if type == plot_type.POINT:
#         #描点
#        # y1 = signal[np.asanyarray(start_x,dtype = np.int)]
#        # y2 = signal[np.asanyarray(end_x,dtype = np.int)]
#         y1 = np.ones((len(start_x))) * np.mean(signal)
#         y2 = np.ones((len(end_x))) * np.mean(signal)
#         plt.plot(start_x,y1,'g*')
#         plt.plot(end_x,y2,'k*')
#     elif type == plot_type.LINE:#划线
    _min = np.min(signal)
    _max = np.max(signal)

    for i in range(len(start_x)):
        plt.plot([start_x[i],start_x[i]],[_min,_max],'r')
        
    for i in range(len(end_x)):
        plt.plot([end_x[i],end_x[i]],[_min,_max],'g')
#     else:  #画长方形
#         if color is None:
#             color = 'g'
#         _min = np.min(signal)
#         _max = np.max(signal)
#         ax = plt.gca()
#         for s,e in zip(start_x, end_x):
#             rect = patches.Rectangle((s, _min), e - s, _max - _min, facecolor=color, alpha=0.5)
#             ax.add_patch(rect) 



def evaluate(r_ref, r_ans,sig_lens =None,fs = 6, thr=30):
    thr_ = thr
    fs_ = fs
    all_TP = 0
    all_FN = 0
    all_FP = 0

    errors = []
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0
  #      if len(r_ref[i]) == 0:
  #          FP += len(r_ans[i])
            
  #      print( r_ref[i],r_ans[i])
        detect_loc = 0
        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_*fs_)[0]
            detect_loc += len(loc)

            
#             if j == 0: 
#               #  err = []
#                 err = np.where((r_ans[i] >= 30*fs_ + thr_*fs_) & (r_ans[i] <= r_ref[i][j] - thr_*fs_))[0]
#             elif j == len(r_ref[i])-1:
#                 err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= sig_lens[i]-30*fs - thr_*fs_))[0]
#             else:
#                 err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= r_ref[i][j+1]-thr_*fs_))[0]

 #           FP = FP + len(err)
            if len(loc) >= 1:
                
                TP += 1
                FP = FP + len(loc) - 1
                
                diff = r_ref[i][j] - r_ans[i][loc[0]]
                errors.append(diff/fs)
                
            elif len(loc) == 0:
                FN += 1
        FP = FP+(len(r_ans[i])-detect_loc)
        
        all_FP += FP
        all_FN += FN
        all_TP += TP
    if all_TP == 0:
        Recall = 0
        Precision = 0
        F1_score = 0
        
    else:
        Recall = all_TP / (all_FN + all_TP)
        Precision = all_TP / (all_FP + all_TP )
        F1_score = 2 * Recall * Precision / (Recall + Precision)
    if all_FP == 0:
        error_rate = 0
    else:
        error_rate =  all_FP / (all_FP + all_TP)
    print("TP's:{} FN's:{} FP's:{}".format(all_TP,all_FN,all_FP))
    #print("正确率：{}，错误率：{}".format(recall,)
    print("正确率:{}, 错误率:{}, F1-Score:{}".format(Recall,error_rate,F1_score))
    print("error mean+std:{}+{},abs_mean+std:{}+{}".format(np.mean(errors), np.std(errors), np.mean(np.abs(errors)), np.std(errors)))
    
def show_features_msg(features, Y, **options):
    '''
    显示特征集的信息，如样本数，正例样本，负例样本。只使用 于二分类样本
    '''
    k  = options.pop('k', 2)
    if k == 2:
        p_feature = features[np.argwhere(Y == 1).flatten(),:] #正例样本
        n_feature = features[np.argwhere(Y == 0).flatten(),:] #负例样本
        size = features.shape[0]
        p_size = p_feature.shape[0]
        n_size = n_feature.shape[0]
        print('总样本数：' + str(size)+ ', 正例样本：' + str(p_size) + ', 负例样本：' + str(n_size)+'，特征数量：' + str(features.shape[1]))
    else:
        print('总样本数：' + str(len(Y)), end=' ')
        for i in range(k):
            size = len(np.argwhere(Y == i))
            print('第' + str(i) + '类' +': '+str(size), end=' ')
        print('特征数量:' + str(features.shape[1]))

def calc_pvalue(arr, Y):
    a1 = []
    a2 = []
    
    for i in range(len(Y)):
        if Y[i] == 1:
            a1.append(arr[i])
        else:
            a2.append(arr[i])    
    import scipy.stats as ss
    stat, p = ss.ranksums(np.array(a1), np.array(a2))
    
    return p   
    
