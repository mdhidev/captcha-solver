#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
from PIL import Image
from joblib import dump, load
# from sklearn_porter import port, save, make, test

dataset_count = {}
imgs = []
labels = []
for subdir, dirs, files in os.walk("./dataset3"):
    for file in files:
      if file != ".DS_Store":
        img = Image.open(os.path.join(subdir, file)).resize(
            (32, 32), Image.LANCZOS).convert("1")
        # print(np.array(img).reshape((-1, 64)).flatten())
        imgs.append(np.array(img).reshape((1024, -1)).flatten())
        labels.append(subdir[-1])
        if subdir[-1] in dataset_count:
          dataset_count[subdir[-1]] += 1
        else:
          dataset_count[subdir[-1]] = 1
        
print(dataset_count)
        
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, imgs, labels):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)
      
# clf = LogisticRegression(max_iter=400)
clf = KNeighborsClassifier(n_neighbors=1)
# clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
X_train, X_test, y_train, y_test = train_test_split(
    np.array(imgs), np.array(labels), test_size=0.1, random_state=42, stratify=np.array(labels))

#--------- Initiate kNN, train it on the training data, then test it with the test data with k=1

# x = np.array(imgs)

# train = x[:500,:].reshape(-1, 64).astype(np.float32)  # Size = (2500,400)
# test = x[500:1000, :].reshape(-1, 64).astype(np.float32)  # Size = (2500,400)
# # Create labels for train and test data
# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = train_labels.copy()

# knn = cv2.ml.KNearest_create()
# knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
# ret, result, neighbours, dist = knn.findNearest(X_test, k=5)

# matches = result == y_test
# correct = np.count_nonzero(matches)
# accuracy = correct*100.0/result.size
# print(accuracy)

#----------

print(X_train)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

dump(clf, "edu.pkl")
# # 2. Port or transpile an estimator:
# output = port(clf, language='js', template='attached')
# print(output)

# # 3. Save the ported estimator:
# src_path, json_path = save(
#     clf, language='js', template='exported', directory='/tmp')
# print(src_path, json_path)

# # 4. Make predictions with the ported estimator:
# y_classes, y_probas = make(clf, X_test, language='js', template='exported')
# print(y_classes, y_probas)

# # 5. Test always the ported estimator by making an integrity check:
# score = test(clf, X_test, language='js', template='exported')
# print(score)

plt.show()
