# The code is a test for random forest classification, last edited in 2023/11/16 by Wan

# importing necessary packages
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold

# Creating dataset for training, only uses small amount of proton data
datadir = '/home/wsy/KM2A/Xishui/'
npzfiles = glob.glob(datadir + 'rec.all.Proton.*.npz')
'''npzfiles = [datadir+'rec.all.Proton.1.e12_1.e13.0_40.QGSII_flukacern.root.weighted.npz',
            datadir+'rec.all.Proton.1.e13_1.e14.40_70.QGSII_flukacern.root.weighted.npz',
            datadir+ 'rec.all.Proton.1.e14_1.e15.70_75.QGSII_flukacern.root.weighted.npz',
            datadir+'rec.all.Proton.1.e15_1.e16.70_75.QGSII_flukacern.root.weighted.npz',
            datadir+'rec.all.Proton.1.e16_1.e17.70_75.QGSII_flukacern.root.weighted.npz']'''
for i, npzfile in enumerate(npzfiles):
    if i==0:
        test = np.load(npzfile, allow_pickle=True)
        theta = test['theta']
        mask = (theta <= 0.261799387991494) # Quality cuts on training data
        arr1 = test['E'][mask]
        arr2 = test['NhitE'][mask]
        arr3 = test['NhitM'][mask]
        arr4 = arr3/arr2
        arr5 = np.zeros(len(arr1))
    else:
        f = np.load(npzfile, allow_pickle=True)
        theta = f['theta']
        mask = (theta <= 0.261799387991494)
        ene = f['E'][mask]
        ED = f['NhitE'][mask]
        MD = f['NhitM'][mask]
        EMr = MD/ED
        lab1 = np.zeros(len(ene))
        arr1 = np.append(arr1, ene)
        arr2 = np.append(arr2,ED)
        arr3 = np.append(arr3,MD)
        arr4 = np.append(arr4, EMr)
        arr5 = np.append(arr5, lab1)
X_p = np.column_stack((arr1,arr2,arr3,arr4,arr5))


# Creating dataset for training, only uses small amount of Fe data
datadir = '/home/wsy/KM2A/Xishui/'
npzfiles = glob.glob(datadir + 'rec.all.Fe.*.npz')
'''npzfiles = [datadir+'rec.all.Fe.1.e13_1.e14.40_70.QGSII_flukacern.root.weighted.npz',
            datadir+'rec.all.Fe.1.e14_1.e15.40_70.QGSII_flukacern.root.weighted.npz',
            datadir+'rec.all.Fe.1.e15_1.e16.70_75.QGSII_flukacern.root.weighted.npz',
            datadir+'rec.all.Fe.1.e16_1.e17.70_75.QGSII_flukacern.root.weighted.npz']'''
for i, npzfile in enumerate(npzfiles):
    if i==0:
        test = np.load(npzfile, allow_pickle=True)
        theta = test['theta']
        mask = theta <= 0.261799387991494
        b1 = test['E'][mask]
        b2 = test['NhitE'][mask]
        b3 = test['NhitM'][mask]
        b4 = b3/b2
        b5 = 1 + np.zeros(len(b1))
    else:
        f = np.load(npzfile, allow_pickle=True)
        theta = f['theta']
        mask = theta <= 0.261799387991494
        eneb = f['E'][mask]
        EDb = f['NhitE'][mask]
        MDb = f['NhitM'][mask]
        EMrb = MDb/EDb
        lab1b = 1 + np.zeros(len(eneb))
        b1 = np.append(b1, eneb)
        b2 = np.append(b2,EDb)
        b3 = np.append(b3,MDb)
        b4 = np.append(b4, EMrb)
        b5 = np.append(b5, lab1b)
X_f = np.column_stack((b1,b2,b3,b4,b5))

# Produce the final dataset for training, and shuffling is necessary. X_A is the total set of features and Y_A is the total set of labels
X_A = np.row_stack((X_p, X_f))
np.random.shuffle(X_A)
Y_A= X_A[:,4:].flatten()
X_A = X_A[:,1:4]

# Split the features and labels into a training set and a testing set, and then use random forest classifier
# K-Fold method can also be used to improve the result
cv = StratifiedKFold(n_splits = 3)
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
fig, ax = plt.subplots()
#X_train, X_test, y_train, y_test = train_test_split(X_A, Y_A, test_size = 0.30)
numTrees = 100
clf = RandomForestClassifier(n_estimators = numTrees,  min_samples_leaf=5)

for i, (train,test) in enumerate(cv.split(X_A,Y_A)):
    clf.fit (X_A[train], Y_A[train])
    y_pred = clf.predict(X_A[test])
    y_score = clf.predict_proba(X_A[test])[:, 1]
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(Y_A[test], y_pred))
    
    # Confusion matrix
    print('Confusion Matrix:', metrics.confusion_matrix(Y_A[test], y_pred))
    met = metrics.confusion_matrix(Y_A[test], y_pred)
    col=['Proton(pred)','Iron(pred)']
    row = ['Proton(real)','Iron(real)']
    vals = met
    plt.figure(figsize = (8,6))
    tab = plt.table(cellText = vals,
                colLabels = col,
                rowLabels = row,
                loc = 'center',
                cellLoc = 'center',
                rowLoc = 'center')
    tab.scale(1,2)
    plt.axis('off')

    # feature importance
    Y = clf.feature_importances_
    X = ['NhitE','NhitM','MD/ED']
    plt.figure(figsize=(12,9))
    Ba = plt.bar(X,Y)
    plt.xlabel("Features", size=20)
    plt.ylabel("Importance", size=20)
    plt.xticks(fontproperties='Times New Roman', size = 15)
    plt.yticks(fontproperties='Times New Roman', size = 15)

   # Show the values of feature importance on the bars 
    for a, b in zip(X, Y):
        plt.text(a, b,'%.2f'%b, ha='center', va='bottom')
    plt.show()


    


    # Ploting the ROC curve
    #auc_score = roc_auc_score(Y_A[test], y_score)
    fpr, tpr, thresholds = roc_curve(Y_A[test], y_score)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
