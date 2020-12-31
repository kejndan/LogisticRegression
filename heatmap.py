import numpy as np
import matplotlib.pyplot as plt
from core import *
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
np.random.seed(100)

# Генерация данных
data_size = 100
x = np.linspace(0, 1, data_size)
y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
x1 = np.linspace(1, 1.5, data_size)
e1 = 10 * np.random.randn(data_size)
e2 = 10 * np.random.randn(data_size)
y1 = y + e1
t1_true = np.zeros((data_size, 3))
t1_true[:,0]=1
y2 = y + e2 + 80
t2_true = np.zeros((data_size, 3))
t2_true[:,1]=1
t3_true = np.zeros((data_size, 3))
t3_true[:,2]=1
y3 = y + e1


# Соединяем наши выборки
res_x = np.hstack((x, x, x1))
res_y = np.hstack((y1, y2, y3))
all_data = np.vstack((res_x, res_y)).T
# plt.plot(all_data[:100,0],all_data[:100,1],'ro')
# plt.plot(all_data[100:200,0],all_data[100:200,1],'go')
# plt.plot(all_data[200:,0],all_data[200:,1],'bo')
# plt.show()

all_data = np.concatenate((all_data,np.zeros(300)[:, np.newaxis]),axis=1)

for i in range(300):
    all_data[i,2] = i // 100

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(
    all_data[:,:2], all_data[:,2].astype(np.int))
# max_train = train_data.max()
# min_train = train_data.min()
# train_data = normalization(train_data, max_train, min_train)
# validation_data = normalization(validation_data, max_train, min_train)
# test_data = normalization(test_data, max_train, min_train)
classifier = SoftmaxRegression(3)
classifier.fit(train_data, train_labels, iter=25000,val_x=validation_data, val_y=validation_labels, lamb=1,step=1e-4)
# classifier.load_weights()
print(classifier.accuracy(test_data,test_labels))
print(classifier.confusion_matrix(test_data,test_labels))
classifier.precision_and_recall()
x, y = np.meshgrid(all_data[:,0],all_data[:,1])

predicted = classifier.predict(np.c_[x.ravel(),y.ravel()])


plt.pcolormesh(x,y, predicted.reshape(x.shape),cmap=ListedColormap(['#FFAAAA', '#AAFFAA','#AAAAFF']))
plt.plot(all_data[:100,0],all_data[:100,1],'ro')
plt.plot(all_data[100:200,0],all_data[100:200,1],'go')
plt.plot(all_data[200:,0],all_data[200:,1],'bo')
patch1 = mpatches.Patch(color='#FFAAAA', label='Class 1')
patch2 = mpatches.Patch(color='#AAFFAA', label='Class 2')
patch3 = mpatches.Patch(color='#AAAAFF', label='Class 3')
plt.legend(handles=[ patch1, patch2, patch3])
plt.title('Num iter: 25000')
plt.show()



