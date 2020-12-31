import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

def softmax(vec):
    exps = np.exp(vec - vec.max(axis=1)[:, np.newaxis])
    return exps/np.sum(exps, axis=1)[:, np.newaxis]

def in_one_hot_encoding(vec):
    targets = np.zeros((len(vec), len(set(vec))))
    mapping = dict(zip(set(vec), np.arange(len(set(vec)))))
    for i in range(len(vec)):
        targets[i, mapping[vec[i]]] = 1
    return targets, mapping

def out_one_hot_encoding(targets, mapping):
    vec = np.zeros(len(targets))
    revers_mapping = (zip(mapping.values(), mapping.keys()))
    for i in range(len(targets)):
        vec[i] = revers_mapping[targets[i].argmax()]
    return vec

def normalization(data, min_val=None, max_val=None):
    min_val = min_val if min_val is not None else min(data)
    max_val = max_val if max_val is not None else max(data)
    return 2*(data - min_val)/(max_val - min_val) - 1


def standardization(data, avg=None, std=None):
    avg = avg if avg is not None else np.average(data)
    std = std if std is not None else data.std()
    return (data - avg)/std


def train_val_test_split(data, labels):
    """
    Делит выборку на обучающий и тестовый датасет
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    mask = np.arange(len(labels))
    len_train = int(0.8 * len(mask))
    len_valid = int(0.1 * len(mask))
    imbalanced = True
    while imbalanced:
        np.random.shuffle(mask)
        train = Counter(labels[mask][:len_train])
        val = Counter(labels[mask][len_train :len_train + len_valid])
        test = Counter(labels[mask][len_train + len_valid :])
        for i in range(len(train.keys())):
            for j in range(len(train.keys())):
                if abs(train[i]-train[j])/train[i] <= 0.1 and\
                abs(val[i]-val[j])/val[j] <= 0.1 and  abs(test[i]-test[j])/test[i] <= 0.1:
                    imbalanced = False

    return data[mask][:len_train], labels[mask][:len_train],\
    data[mask][len_train :len_train + len_valid], labels[mask][len_train :len_train + len_valid],\
    data[mask][len_train + len_valid :], labels[mask][len_train + len_valid :]


class SoftmaxRegression :
    def __init__(self, number_classes) :
        self.number_classes = number_classes
        self.weights = None
        self.biases = None
        self.features = None
        self.target = None

    def get_gradient(self, x, y_true, y_pred, lamb) :
        weights = np.dot((y_pred - y_true).T, x) + lamb * self.weights / x.shape[0]
        biases = (y_pred - y_true).sum(axis=0)
        mean_weights = weights/x.shape[0]
        mean_biases = biases/x.shape[0]
        return mean_weights, mean_biases

    def loss(self, y_true, y_pred) :
        k = np.max(y_pred, axis=1)[:, np.newaxis]
        q = (y_pred - k - np.log(np.exp(y_pred - k).sum(axis=1))[:, np.newaxis])
        loss_func = -y_true * q
        return loss_func.sum()

    def predict(self, X) :
        probability_scores = np.argmax(softmax(X @ self.weights.T + self.biases), axis=1)
        return probability_scores

    def BGD(self, X, y, iterations, batch_size, step=1e-4, init_weights='Xavier', graphs=True,**kwargs) :
        if init_weights == 'Xavier' :
            self.weights = np.array(
                [2 / self.number_classes + len(X[0]) for i in range(self.number_classes * len(X[0]))]).reshape(
                (self.number_classes, len(X[0])))
            self.biases = self.biases = np.array(
                [2 / self.number_classes + len(X[0]) for i in range(self.number_classes)])
        elif init_weights == 'large_modulus' :
            self.weights = ((2 * np.random.random(self.number_classes * len(X[0])) - 1) * 10 ** 3).reshape(
                (self.number_classes, len(X[0])))
            self.biases = (2 * np.random.random(self.number_classes) - 1) * 10 ** 3
        elif init_weights == 'small' :
            self.weights = ((1 / (self.number_classes * len(X[0])) * np.random.random(
                self.number_classes * len(X[0])) - 1 / (self.number_classes * len(X[0])))).reshape(
                (self.number_classes, len(X[0])))
            self.biases = 1 / self.number_classes * np.random.random(self.number_classes) - 1 / self.number_classes

        Q = self.loss(y, X @ self.weights.T + self.biases) / len(X)
        loss_data = []
        accuracy_data = []
        max_accuracy = 0
        for iter in range(1, iterations + 1) :
            num_samples = [random.randint(0, len(X) - 1) for s in range(batch_size)]
            X_samples = X[num_samples]
            y_samples = y[num_samples]
            new_weights, new_biases = self.get_gradient(X_samples, y_samples,
                                                        softmax(X_samples @ self.weights.T + self.biases), kwargs['lamb'])
            self.weights -= step * new_weights
            self.biases -= step * new_biases
            e = self.loss(y_samples, X_samples @ self.weights.T + self.biases)
            Q = e / iter + (1 - 1 / iter) * Q

            if 'val_x' in kwargs and 'val_y' in kwargs and iter % 100 == 0 :
                print(f'{iter} {Q} Accuracy_val: {self.accuracy(kwargs["val_x"], kwargs["val_y"])}  '
                      f'Accuracy_train :{self.accuracy(self.features, self.target)}')
                if self.accuracy(kwargs["val_x"], kwargs["val_y"]) > max_accuracy :
                    max_accuracy = self.accuracy(kwargs["val_x"], kwargs)
                    self.save_weights()
                accuracy_data.append(self.accuracy(kwargs["val_x"], kwargs["val_y"]))
            loss_data.append(Q)
        if graphs:
            plt.plot(np.arange(1, iterations + 1), loss_data)
            plt.title(f"Loss. Lamb:{kwargs['lamb']}")
            plt.xlabel('Number_of_iteration')
            plt.ylabel('Loss')
            plt.show()
            plt.plot(np.arange(1, iterations + 1, 100), accuracy_data)
            plt.xlabel('Number_of_iteration')
            plt.ylabel('Accuracy')
            plt.title(f"Accuracy(valid). Lamb:{kwargs['lamb']}")
            plt.show()

    def fit(self, X, y, iter=45000, batch_size=32, **kwargs):
        if 'lamb' in kwargs :
            lamb = kwargs['lamb']
        else :
            lamb = 1
        if 'init_weights' in kwargs:
            init_weights = kwargs['init_weights']
        else:
            init_weights = 'Xavier'
        if 'step' in kwargs:
            step = kwargs['step']
        else:
            step = 1e-4
        self.features = X
        self.target = y
        t, mapping = in_one_hot_encoding(y)
        if 'val_x' in kwargs and 'val_y' in kwargs :
            self.BGD(X, t, iter, batch_size, init_weights=init_weights,
                     val_x=kwargs['val_x'], val_y=kwargs['val_y'], lamb=lamb, step=step)
        else :
            self.BGD(X, t, iter, batch_size, lamb=lamb)


    def accuracy(self, data, labels) :
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        probability_scores = np.argmax(softmax(data @ self.weights.T + self.biases),axis=1)
        return (probability_scores == labels).mean()

    def confusion_matrix(self, data, labels) :
        """
        Данная функция вычисляет confusion_matrix на основании переданных данных
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        results =  np.argmax(softmax(data @ self.weights.T + self.biases),axis=1)
        conf_matrix = [[0] * self.number_classes for _ in range(self.number_classes)]
        for i in range(len(results)) :
            conf_matrix[results[i]][labels[i]] += 1
        self.conf_matrix = np.array(conf_matrix)
        return self.conf_matrix

    def precision_and_recall(self) :
        """
        Данная функция вычисляет precision и recall на основании confusion matrix
        :return:
        """
        for i in range(self.number_classes) :
            if self.conf_matrix[i, :].sum() != 0 :
                precision = self.conf_matrix[i, i] / self.conf_matrix[i, :].sum()
            else :
                precision = 0
            assert self.conf_matrix[:, i].sum() != 0, f'Данные не содержат класс {i}'
            recall = self.conf_matrix[i, i] / self.conf_matrix[:, i].sum()
            print(f'Digit:{i} Precision:{precision} Recall:{recall}')

    def save_weights(self) :
        with open('weights.pkl', 'wb') as f :
            pickle.dump(np.concatenate((self.weights, self.biases[:, np.newaxis]), axis=1), f)

    def load_weights(self) :
        with open('weights.pkl', 'rb') as f :
            weights = pickle.load(f)
            self.weights, self.biases = weights[:, :-1], weights[:, -1]