import load_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


np.random.seed(100)
# return DataSet class
data = load_data.read_data_sets(one_hot=True)

# get train data and labels by batch size
train_x, train_label = data.train.next_batch(84420)

# get test data
test_x = data.test.data

# get test labels
test_labels = data.test.labels
# get sample number
n_samples = data.train.num_examples
# use knn for classification
knn = KNeighborsClassifier(n_neighbors= 3)
# train the model
knn.fit(train_x, train_label)
# predict the values
y_pred = knn.predict(test_x)
# print accuracy
print("Accuracy" , accuracy_score(y_pred, test_labels)*100)
