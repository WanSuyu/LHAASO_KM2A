# The code is for showing the scattering plot of proton and iron events in the parameter space of (NED,NMD,NED/NMD)
# Last edited 2023/12/1
import numpy as np
import matplotlib.pyplot as plt
import glob

# Reading Proton data
datadir = '/home/wsy/KM2A/Xishui/'
npzfiles = glob.glob(datadir + 'rec.all.Proton.*.npz')
for i, npzfile in enumerate(npzfiles):
    if i==0:
        test = np.load(npzfile, allow_pickle=True)
        theta = test['theta']
        mask = (theta <= 0.261799387991494)
        arr1 = test['NhitE'][mask]
        arr2 = test['NhitM'][mask]
        arr3 = arr1/arr2
        arr4 = np.zeros(len(arr1))
    else:
        f = np.load(npzfile, allow_pickle=True)
        theta = f['theta']
        mask = (theta <= 0.261799387991494)
        ED = f['NhitE'][mask]
        MD = f['NhitM'][mask]
        emr = ED/MD
        lab1 = np.zeros(len(ED))
        
        arr1 = np.append(arr1,ED)
        arr2 = np.append(arr2,MD)
        arr3 = np.append(arr3, emr)
        arr4 = np.append(arr4, lab1)
X_p = np.column_stack((arr1,arr2,arr3,arr4))

# Reading Fe data
datadir = '/home/wsy/KM2A/Xishui/'
npzfiles = glob.glob(datadir + 'rec.all.Fe.*.npz')
for i, npzfile in enumerate(npzfiles):
    if i==0:
        test = np.load(npzfile, allow_pickle=True)
        theta = test['theta']
        mask = (theta <= 0.261799387991494)
        b1 = test['NhitE'][mask]
        b2 = test['NhitM'][mask]
        b3 = b1/b2
        b4 = 1 + np.zeros(len(b1))
    else:
        f = np.load(npzfile, allow_pickle=True)
        theta = f['theta']
        mask = (theta <= 0.261799387991494)
        EDb = f['NhitE'][mask]
        MDb = f['NhitM'][mask]
        emrb = EDb/MDb
        lab2 = 1 + np.zeros(len(emrb))
        
        b1 = np.append(b1,EDb)
        b2 = np.append(b2,MDb)
        b3 = np.append(b3, emrb)
        b4 = np.append(b4, lab2)
X_f = np.column_stack((b1,b2,b3,b4))

X_A = np.row_stack((X_p, X_f))
np.random.shuffle(X_A)

# plotting the scattering plot of all proton and iron events
x = X_A[:,0].flatten()
y = X_A[:,1].flatten()
z = X_A[:,2].flatten()
typ = X_A[:,3].flatten()
clr=[]
for j in range(0,len(typ)):
    if typ[j]==0:
        clr.append('r')
    else:
        clr.append('g')
fig = plt.figure(dpi=720)
plt.style.use('default')
ax = fig.add_subplot(projection = '3d')
ax.set_title('All_events')
ax.scatter(x, y, z, c = clr)
ax.set_xlabel('NED')
ax.set_ylabel('NMD')  
ax.set_zlabel('ED/MD')

plt.figure(dpi=1024)
plt.style.use('default')
plt.suptitle('Projections on different planes')
plt.subplot(1,3,1)
plt.title('ED-MD')
plt.scatter(x, y, c = clr, s = 1)
plt.subplot(1,3,2)
plt.title('ED-EM_ratio')
plt.scatter(x, z, c = clr, s = 1)
plt.subplot(1,3,3)
plt.title('MD-EM_ratio')
plt.scatter(y, z, c = clr, s = 1)