
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable


class KNeighborsClassifier:

    def __init__(self, n_neighbors=5, distance_metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for feature in range(len(X_test)):
            label = self.findNearestNeighbor(X_test[feature, :])[0]
            # adds the predicted label into the predictions array
            predictions.append(label)
        return predictions

    def euclidean_distance(self, X_train, feature):
        return np.sqrt(np.sum((np.array(X_train) - np.array(feature))**2))

    def manhattan_distance(self, u, v):
        manhattan_distance = np.sum(np.abs(np.array(u) - np.array(v)))
        return manhattan_distance

    def cosine_distance(self, u, v):
        uv = np.average(u * v,)
        uu = np.average(np.square(u))
        vv = np.average(np.square(v))
        cosine_distance = 1.0 - uv / np.sqrt(uu * vv)
        return cosine_distance

    def findNearestNeighbor(self, feature):

        distances = []
        labels = []
        if self.distance_metric == 'euclidean':
            for i in range(len(self.X_train)):
                distances.append(self.euclidean_distance(self.X_train[i, :], feature))
                labels.append(str(self.y_train[i]))

        elif self.distance_metric == 'manhattan':
            for i in range(len(self.X_train)):
                distances.append(self.manhattan_distance(self.X_train[i, :], feature))
                labels.append(str(self.y_train[i]))

        elif self.distance_metric == 'cosine':
            for i in range(len(self.X_train)):
                distances.append(self.cosine_distance(self.X_train[i, :], feature))
                labels.append(str(self.y_train[i]))

        else:
            print("Non-valid distance metric!!!\nDistance metrics : "
                  "'euclidean', 'manhattan', 'cosine'")
            return

        sorted_list = sorted(zip(distances, labels), reverse=False)[:self.n_neighbors]
        most_nearest_labels = list(zip(*sorted_list))[1]
        # counts elements in nearest labels array
        count = Counter(most_nearest_labels)
        return count.most_common()[0]  # return the most common label

    def accuracy(self, y_test, y_predicted):
        correct_prediction = 0
        wrong_prediction = 0
        total_prediction = 0
        for i in range(len(y_test)):
            if str(y_test[i]) == y_predicted[i]:
                correct_prediction += 1
            else:
                wrong_prediction += 1
            total_prediction += 1
        accuracy = (correct_prediction / float(total_prediction)) * 100.0
        return (self.n_neighbors, accuracy, wrong_prediction)

    def draw_decision_boundaries(self):

        """ ==============Visualize decision boundaries===============
        For decision boundaries we will find each point on
        coordinate space, then we will predict the label
        for each point. After predicting all points, we will
        create space color list, then we map each point
        in the prediction list according to its label.
        Finally, we will draw this points on scatter plot
        and add training points on this plot.
        ==============================================================
        """
        # Find each point on coordinate space
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        h = .05  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.reshape(Z, xx.shape)

        space_color_list = []  # create color list for space

        # map each point on coordinate to a color according its label
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):

                if str(Z[i][j]) == "['Iris-setosa']":
                    space_color_list.append('#FFDDDD')

                elif str(Z[i][j]) == "['Iris-versicolor']":
                    space_color_list.append('#DDFFDD')

                else:
                    space_color_list.append('#FFFFDD')

        plt.figure()

        # plot each point on space
        plt.scatter(xx, yy, marker='o', c=space_color_list)

        # Plot also the training points
        plt.scatter(self.X_train[:, [0]], self.X_train[:, [1]], marker='o', c=colors)

        plt.title('Decision Boundaries\n(k = {}  Distance = {})'.format(self.n_neighbors,
                                                                        self.distance_metric))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 4')
        plt.xlim(self.X_train[:, 0].min() - 0.5, self.X_train[:, 0].max() + 0.5)
        plt.ylim(self.X_train[:, 1].min() - 0.5, self.X_train[:, 1].max() + 0.5)
        plt.draw()



filename = 'iris_data.txt'

# assign dataframe to the variable : data
data = pd.read_csv(filename, sep=',', header=None)

# grouped values by their class label
grouped_data = data.groupby([4])

# transform DataFrameGroupBy object to DataFrame
grouped_data_df = grouped_data.apply(lambda x: x)


# for training we select first 30 row of each group in our dataset
# and last 20 rows of each group for testing
train_rows = [i for i in range(150) if ((i < 30) or (80 > i > 49) or (130 > i > 99))]
test_rows = [i for i in range(150) if (not((i < 30) or (80 > i > 49) or (130 > i > 99)))]
X_train = np.array(grouped_data_df.iloc[train_rows, [0, 3]])
y_train = np.array(grouped_data_df.iloc[train_rows, [4]])
X_test = np.array(grouped_data_df.iloc[test_rows, [0, 3]])
y_test = np.array(grouped_data_df.iloc[test_rows, [4]])

"""
Create a color list for scatter plot
red for Iris-setosa, green for Iris-versicolor
and yellow for Iris-virginica
"""
colors = []
for i in range(len(y_train)):
    if y_train[i, 0] == 'Iris-setosa':
        colors.append('r')
    elif y_train[i, 0] == 'Iris-versicolor':
        colors.append('g')
    else:
        colors.append('y')

# Visualize training data with scatter plot
plt.figure(1)
plt.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=colors)
plt.xlabel('Feature 1')
plt.ylabel('Feature 4')
plt.title('Training Set')
plt.draw()


"""===============Creating Accuracy and Error Count Tables===============

For creating tables we will use prettyTable library.
Output for each distance metric given as folowing.

+----------+--------------------+-----------------------+
| k Number | Euclidean Accuracy | Euclidean Error Count |
+----------+--------------------+-----------------------+
|    1     | 93.33333333333333  |           4           |
|    3     | 96.66666666666667  |           2           |
|    5     | 96.66666666666667  |           2           |
|    7     | 96.66666666666667  |           2           |
|    9     | 96.66666666666667  |           2           |
|    11    | 96.66666666666667  |           2           |
|    13    | 96.66666666666667  |           2           |
|    15    | 96.66666666666667  |           2           |
+----------+--------------------+-----------------------+


+----------+--------------------+-----------------------+
| k Number | Manhattan Accuracy | Manhattan Error Count |
+----------+--------------------+-----------------------+
|    1     | 93.33333333333333  |           4           |
|    3     | 96.66666666666667  |           2           |
|    5     | 96.66666666666667  |           2           |
|    7     | 96.66666666666667  |           2           |
|    9     | 96.66666666666667  |           2           |
|    11    | 96.66666666666667  |           2           |
|    13    | 96.66666666666667  |           2           |
|    15    |        95.0        |           3           |
+----------+--------------------+-----------------------+


+----------+-------------------+--------------------+
| k Number |  Cosine Accuracy  | Cosine Error Count |
+----------+-------------------+--------------------+
|    1     | 86.66666666666667 |         8          |
|    3     | 91.66666666666666 |         5          |
|    5     | 88.33333333333333 |         7          |
|    7     | 88.33333333333333 |         7          |
|    9     | 91.66666666666666 |         5          |
|    11    | 91.66666666666666 |         5          |
|    13    | 88.33333333333333 |         7          |
|    15    | 88.33333333333333 |         7          |
+----------+-------------------+--------------------+

"""

k_numbers_list = [ i for i in range(16) if i%2 == 1]  # Creates odd k number from 1 to 15

euclidean_results = []
manhattan_results = []
cosine_results = []

# calculate accuracy and error count for each k and euclidean distance
for i in k_numbers_list:
        clf = KNeighborsClassifier(i, 'euclidean')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        euclidean_results.append(clf.accuracy(y_test, predictions))

# calculate accuracy and error count for each k and manhattan distance
for i in k_numbers_list:
        clf = KNeighborsClassifier(i, 'manhattan')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        manhattan_results.append(clf.accuracy(y_test, predictions))

# calculate accuracy and error count for each k and cosine distance
for i in k_numbers_list:
        clf = KNeighborsClassifier(i, 'cosine')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        cosine_results.append(clf.accuracy(y_test, predictions))

euc_table = PrettyTable()
euc_table.field_names = ["k Number", "Euclidean Accuracy", "Euclidean Error Count"]

# fill values for Euclidean table
for index in range(len(euclidean_results)):
    euc_table.add_row(euclidean_results[index])


man_table = PrettyTable()
man_table.field_names = ["k Number", "Manhattan Accuracy", "Manhattan Error Count"]

# fill values for Manhattan table
for index in range(len(manhattan_results)):
    man_table.add_row(manhattan_results[index])

cos_table = PrettyTable()
cos_table.field_names = ["k Number", "Cosine Accuracy", "Cosine Error Count"]

# fill values for Cosine table
for index in range(len(cosine_results)):
    cos_table.add_row(cosine_results[index])

print(euc_table)   # Print Euclidean table
print('\n')
print(man_table)  # Print Manhattan table
print('\n')
print(cos_table)  # Print Cosine table

"""
================ Visualize specific decision boundaries =================
"""

# create an instance of KNeighborsClassifier for k = 3, euclidean distance
clf = KNeighborsClassifier(3, distance_metric='euclidean')
# and fit the training data
clf.fit(X_train,y_train)
# draw the decision boundaries
clf.draw_decision_boundaries()

# create an instance of KNeighborsClassifier for k = 3, euclidean distance
clf = KNeighborsClassifier(3, distance_metric='manhattan')
# and fit the training data
clf.fit(X_train,y_train)
# draw the decision boundaries
clf.draw_decision_boundaries()

# create an instance of KNeighborsClassifier for k = 3, cosine distance
clf = KNeighborsClassifier(3, distance_metric='cosine')
# fit the training data
clf.fit(X_train,y_train)
# draw the decision boundaries
clf.draw_decision_boundaries()

# create an instance of KNeighborsClassifier for k = 1, euclidean distance
clf = KNeighborsClassifier(1, distance_metric='euclidean')
# fit the training data
clf.fit(X_train,y_train)
# the decision boundaries
clf.draw_decision_boundaries()
# show plots
plt.show()
