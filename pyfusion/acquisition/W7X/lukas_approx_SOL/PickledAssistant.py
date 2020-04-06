# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:22:41 2016

@author: luru

Probe number lookup assistant
"""

import pickle
import datetime as dt
import numpy as np
import os 
#global I_array,U_array

def loadglobal():
    global I_array,U_array
    filepath=os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(filepath,'I_array.pickle'),'rb') as a:
        I_array=pickle.load(a)
    with open(os.path.join(filepath,'U_array.pickle'),'rb') as b:
        U_array=pickle.load(b)



def lookupChannel(probeNumber,divertor,signal,date):
    '''
    given the probe tip number, 0 or 1 for lower and upper divertor, the date and 0 or 1 for 'U' or 'I', this function returns a tuple with codac Station and channel 
    '''
    filepath=os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(filepath,'I_array.pickle'),'rb') as a:
        I_array=pickle.load(a)
    with open(os.path.join(filepath,'U_array.pickle'),'rb') as b:
        U_array=pickle.load(b)

    divd={'lower':0,'AFE50':0, 'Lower':0, 'LOWER':0, 0:0, 'upper':1, 'Upper':1, 'UPPER':1, 'AFE51':1, 1:1}
    sigd={'U':0,'Spannung':0,'Voltage':0,'x':0,0:0,'I':1,'Strom':1,'Current':1,'y':1, 1:1}
    if not sigd[signal]:
        return U_array[findIndex(date)][probeNumber-1+(divd[divertor]*20)]
    if sigd[signal]:
        return I_array[findIndex(date)][probeNumber-1+(divd[divertor]*20)]

def testfunc():
    return (1, 2)

def testfuncB(l):
    print(type(l))
    return 1

def lookupPosition(probeNumber,diagnostic):
    '''returns the position of a probe in mm global W7x Coordinates, and distance from lcfs'''
    diagD={'lower':0,'AFE50':0, 0:0, 'upper':1, 'AFE51':1, 1:1, 'mirnov':2, 2:2}
    pos=[]
    #lower limiter
    pos.append(np.array([[  1,   1.80290000e+03,  -5.37900000e+03,  -2.04600000e+02,17.8],
       [  2,   1.69700000e+03,  -5.40030000e+03,         -2.01900000e+02,22.3],
       [  3,   1.79850000e+03,  -5.38910000e+03,         -1.83900000e+02,12.6],
       [  4,   1.79370000e+03,  -5.38530000e+03,         -2.24800000e+02,10.3],
       [  5,   1.78750000e+03,  -5.39310000e+03,         -2.16800000e+02,5.9],
       [  6,   1.78640000e+03,  -5.39650000e+03,         -2.05100000e+02,5.0],
       [  7,   1.78400000e+03,  -5.40500000e+03,         -1.69400000e+02,3.28],
       [  8,   1.78080000e+03,  -5.40220000e+03,         -1.96200000e+02,2.43],
       [  9,   1.77670000e+03,  -5.39990000e+03,         -2.19500000e+02,1.46],
       [  10,   1.77090000e+03,  -5.40500000e+03,         -2.07900000e+02,0.0879],
       [  11,   1.75650000e+03,  -5.40590000e+03,         -2.21400000e+02,0.0],
       [  12,   1.75670000e+03,  -5.41110000e+03,         -1.92900000e+02,0.0275],
       [  13,   1.74460000e+03,  -5.41120000e+03,         -2.07400000e+02,0.288],
       [  14,   1.73940000e+03,  -5.41410000e+03,         -1.92300000e+02,1.5],
       [  15,   1.72990000e+03,  -5.40720000e+03,         -2.28100000e+02,3.16],
       [  16,   1.72610000e+03,  -5.41540000e+03,         -1.78400000e+02,5.85],
       [  17,   1.72210000e+03,  -5.40780000e+03,         -2.18000000e+02,6.39],
       [  18,   1.72060000e+03,  -5.40340000e+03,         -2.39000000e+02,6.32],
       [  19,   1.71500000e+03,  -5.41040000e+03,         -1.91400000e+02,10.8],
       [  20,   1.70540000e+03,  -5.40210000e+03,         -2.17200000e+02,15.5]]))
    #upper limiter
    pos.append(np.array([[1, 1703.1  ,-5411.4,	204.6,17.77],
                    [2, 1801.3  ,-5366.4,	201.9,22.25],
                    [3, 1712.6  ,-5417,		183.9,12.56],
                    [4, 1714.3  ,-5411.2,	224.8,10.35],
                    [5, 1723.9  ,-5413.8,	216.8,5.90],
                    [6, 1726.7  ,-5415.9,	205.1,5.0],
                    [7, 1733.7  ,-5421.3,	169.4,3.28],
                    [8, 1734.7  ,-5417.2	,196.2,2.43],
                    [9, 1736.6  ,-5412.9	,219.5,1.46],
                    [10,1744.3  ,-5413.6      ,207.9,0.09],
                    [11,1756.5  ,-5405.9	,221.4,0],
                    [12,1759.4  ,-5410.3	,192.9,0.28],
                    [13,1769.2  ,-5403.2	,207.4,0.29],
                    [14,1775.1  ,-5402.6	,192.3,1.5],
                    [15,1778.7  ,-5391.3	,228.1,3.16],
                    [16,1786.6  ,-5395.8	,178.4,5.85],
                    [17,1785.4  ,-5387.2	,218,6.39],
                    [18,1784	   ,-5382.8    	,239,6.32],
                    [19,1792.7  ,-5385.1	,191.4,10.78],
                    [20,1795.6  ,-5372.8	,217.2,15.47]]))
    #mirnov coils still missing
    pos.append(np.array([1,2,3]))
    '''This is the only part that does something: diagnostic string->dictionary->positions array->probeNumber->probeCoordinates'''
    return pos[diagD[diagnostic.lower()]][probeNumber-1][1:]
    
def rotatePos(position,angle):
    angle*=np.pi/180
    R=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    position=np.dot(R,position)
    return position
    
def lookupDistance(probeNumber1, probeNumber2,diagnostic,mode='mag'):
    posD={'x':1,'y':2,'z':3,'lcfs':4}
    if mode in 'xyzlfcs':
        return  lookupPosition(probeNumber2,diagnostic)[posD[mode]]-lookupPosition(probeNumber1,diagnostic)[posD[mode]]   
    if mode=='mag':
        return np.sum((lookupPosition(probeNumber2,diagnostic)[:3]-lookupPosition(probeNumber1,diagnostic)[:3])**2)**0.5
    if mode=='all':
        return lookupPosition(probeNumber2,diagnostic)-lookupPosition(probeNumber1,diagnostic)
        
def findIndex(date):
    year,month,day=date
    current=dt.datetime(year,month,day)
    if current<dt.datetime(2016,1,18):
        print('invalid date')
        return None
    elif current<dt.datetime(2016,1,20):
        return 0
    elif current<dt.datetime(2016,1,22):
        return 1
    elif current<dt.datetime(2016,1,28):
        return 2
    elif current<dt.datetime(2016,2,2):
        return 3
    elif current<dt.datetime(2016,2,16):
        return 4
    elif current<dt.datetime(2016,2,17):
        return 5
    elif current<dt.datetime(2016,2,18):
        return 6
    elif current<dt.datetime(2016,2,23):
        return 7
    else:
        return 8        
