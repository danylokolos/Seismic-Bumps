# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:54:14 2021

@author: Danylo
"""

""" Script to test Random Forest Classification on Seismic Bumps Data"""

### Setup
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
le = LabelEncoder()

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)
random_state = 50

### Format Data
# Read in data
df = pd.read_csv('seismic-bumps-data.csv')
# df = pd.read_csv('ValidationTestDatasetBigger.csv')

# Reorganize data
target_names = np.array(list(df['class'].values))
indexes = np.unique(target_names, return_index=True)[1]
target_names = [target_names[index] for index in sorted(indexes)]
target_names = np.array(target_names)

labels = df['class'].values

# Encode Labels (Convert from text to numbers)
#df['seismic'] = df['seismic'].astype('category')

df['seismic'] = le.fit_transform(df['seismic'])
df['seismoacoustic'] = le.fit_transform(df['seismoacoustic'])
df['shift'] = le.fit_transform(df['shift'])
df['ghazard'] = le.fit_transform(df['ghazard'])

# Split into Training and Testing data
train, test, train_labels, test_labels = train_test_split(df,
                                                          labels, 
                                                          stratify = labels,
                                                          test_size=0.2)

# Show the number of observations for the test and training dataframes
print('Number of observations in the entire dataset:', len(df))
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names
features = df.columns[:18]
y  = train.iloc[:,18].values

X_train = train[features]
y_train = train_labels

### Resample minority class using SMOTE
"""
# sudo pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

counter = Counter(y_train)
print('Before',counter)
ratio = 1.0
smt=SMOTE(random_state=random_state, sampling_strategy=ratio,k_neighbors=9)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
counter = Counter(y_train_sm)
print('After',counter)

X_train = X_train_sm
y_train = y_train_sm

"""



### Undersample Majority Class - Random
"""
from imblearn.under_sampling import RandomUnderSampler
counter = Counter(y_train)
print('Before',counter)
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
counter = Counter(y_train_under)
print('After',counter)

X_train = X_train_under
y_train = y_train_under

"""

### Undersample Majority Class - Near Miss
"""
from imblearn.under_sampling import NearMiss

counter = Counter(y_train)
print('Before',counter)

undersample = NearMiss(version=1, n_neighbors=3)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

counter = Counter(y_train_under)
print('After',counter)

X_train = X_train_under
y_train = y_train_under
"""


### Undersample Majority Class - Nearest Neighbour
from imblearn.under_sampling import CondensedNearestNeighbour

counter = Counter(y_train)
print('Before',counter)

undersample = CondensedNearestNeighbour(n_neighbors=1)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

counter = Counter(y_train_under)
print('After',counter)

X_train = X_train_under
y_train = y_train_under



### Create Model
# Create a random forest Classifier
clf = RandomForestClassifier(n_estimators=50, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)
clf.fit(X_train, y_train)
clf.predict(test[features])

# See probabilities of first few test samples
clf.predict_proba(test[features])[0:10]

# Create actual english names for the land types for each predicted land type
preds = target_names[clf.predict(test[features])]
# View the PREDICTED species for the first five observations
preds[0:5]

# View the ACTUAL species for the first five observations
test['class'].head()

# Create confusion matrix
pd.crosstab(test['class'], preds, rownames=['Actual Class'], colnames=['Predicted Class'])

# View a list of the features and their importance scores
list(zip(X_train, clf.feature_importances_))



### Extra Stats
# Fit on training data
model = clf
modelresults = model.fit(X_train, y_train)

n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting)
train_rf_predictions = model.predict(X_train)
train_rf_probs = model.predict_proba(X_train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(test[features])
rf_probs = model.predict_proba(test[features])[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18



### plot ROC Curve for single class
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification

#X_train = X_train
X_test = test[features]
#y_train = y_train
y_test = test_labels

probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('RF_ROCCurve.png')

plt.show()




### Plot ROC Curve for multiclass
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
# X = iris.data
X = df[features]
# y = iris.target
y = labels

# Binarize the output
#y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6])
y = label_binarize(y, classes=[0, 1])


#n_classes = y.shape[1]
n_classes = y.shape[1]


# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

X_train = train[features]
X_test = test[features]
y_train = train_labels
y_test = test_labels
#y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
y_test_bin = label_binarize(y_test, classes=[0, 1])


# Learn to predict each class against the other
classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True, random_state=random_state)
)
# classifier = RandomForestClassifier(n_estimators=50, 
#                                 max_features = 'sqrt',
#                                 n_jobs=-1, verbose = 1)
classifier = clf
y_score = modelresults.predict_proba(X_test)


# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
# plt.figure()
plt.figure(figsize=(10,10))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["grey", "lawngreen", "yellow", "forestgreen", "blue", "white", "black"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        # lw=lw,
        label="ROC curve of {0} (area = {1:0.2f})".format(target_names[i], roc_auc[i]),
    )

# plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.plot([0, 1], [0, 1], "k--")


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristics")
#plt.legend(loc="lower right")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.show()

plt.savefig('RF_ROCCurve.png')
"""



### Plot Confusion Matrix

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = target_names,
                      title = 'Seismic Bumps Confusion Matrix')

plt.savefig('RF_ConfusionMatrix.png')



### Plot Precision-Recall Curve
from sklearn.metrics import PrecisionRecallDisplay

#y_score = clf.decision_function(X_test)
y_score = clf.predict_proba(test[features])
y_score = y_score[:,1]

disp = PrecisionRecallDisplay.from_predictions(y_test, y_score, name="Random Forest")
#disp.plot()
plt.title('Precision Recall Curve')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc = 'upper right')
# plt.legend(loc = 'lower right')

plt.savefig('RF_PrecisionRecallCurve.png')
plt.show()

