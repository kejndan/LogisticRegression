import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from core import *

if __name__ == '__main__':
    digits = datasets.load_digits()


    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(
        digits.data, digits.target)
    max_train = train_data.max()
    min_train = train_data.min()
    train_data = normalization(train_data, max_train, min_train)
    validation_data = normalization(validation_data, max_train, min_train)
    test_data = normalization(test_data, max_train, min_train)
    classifier = SoftmaxRegression(10)
    # classifier.fit(train_data, train_labels, iter=30000,val_x=validation_data, val_y=validation_labels, lamb=30)
    classifier.load_weights()
    print(f'Accuracy:{classifier.accuracy(test_data,test_labels)}')
    print(f'Predicted class: {classifier.predict(test_data[:1])} True class:{test_labels[0]}')



    print(classifier.confusion_matrix(test_data,test_labels))
    classifier.precision_and_recall()







