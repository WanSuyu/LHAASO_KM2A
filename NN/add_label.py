# Obi-Wan 2024/4/23
# All simulations satisfy that: costheta<0.9, 300 <=core_dist <= 500, NhitE>=50
import numpy as np
import glob

datadir = 'D:\LHAASO\LHAASO\Data\Simulation\Released\\'
desdir = 'D:\LHAASO\LHAASO\Data\Simulation\R_copy\\'
pars = ['Proton','He','CNO','MgAlSi','Fe']
for par in pars:
    files = glob.glob(datadir+'label_*'+par+'.npy')
    for i, fi in enumerate(files):
        if i==0:
            cols=np.load(fi,allow_pickle=True)
        else:
            f = np.load(fi,allow_pickle=True)
            cols = np.row_stack((cols,f))
    labels = cols[:,0].astype(np.float32)
    logNED = cols[:,4]
    logNMD = cols[:,5]
    ratio = (10**logNED) / (10**logNMD)
    costheta = cols[:,6]
    if par=='Proton':
        mass = np.zeros(len(labels))+np.log(2)
    elif par =='He':
        mass = np.zeros(len(labels))+np.log(5)
    elif par == 'CNO':
        mass = np.zeros(len(labels))+np.log(15)
    elif par == 'MgAlSi':
        mass = np.zeros(len(labels))+np.log(29)
    elif par == 'Fe':
        mass = np.zeros(len(labels))+np.log(57)
    np.save(desdir+'All_events_'+par+'.npy',np.column_stack((labels,costheta,logNED,logNMD,ratio,mass)))