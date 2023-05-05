# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:02:30 2023

@author: lawashburn
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

tps = pd.read_csv(r"C:\Users\lawashburn\Documents\MSI_test\python_scripts\matched_52panc_52psc_data_sum60ppm_withtypes.csv")
tps.shape
LE = LabelEncoder()
tps['CellType'] = LE.fit_transform(tps['CellType'])
tps=tps.iloc[: , 1:]
tps = tps.astype(float)
# Sample the data - 100k
X, y = tps.drop("CellType", axis=1), tps[["CellType"]].values.flatten()
pipe = make_pipeline(SimpleImputer(strategy="mean"))
X = pipe.fit_transform(X.copy())

manifold = umap.UMAP().fit(X, y)
X_reduced = manifold.transform(X)
X_reduced.shape
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=10);
plt.title("UMAP Reduction")
plt.show()
X_embedded = TSNE(n_components=2, learning_rate='auto',random_state=0).fit_transform(X)
X_embedded.shape

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=10);
plt.title("TSNE Reduction")
plt.show()

pca = PCA(n_components=2)
X_pca =pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=10);
plt.title("PCA Reduction")
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion_matrix(y_test, y_pred)

model2 = MLPClassifier().fit(X_train, y_train)
y_pred3 = model2.predict(X_test)

accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy:", accuracy3)
confusion_matrix(y_test, y_pred3)

model = LogisticRegression().fit(X_train, y_train)
y_pred2= model.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy2)
confusion_matrix(y_test, y_pred2)