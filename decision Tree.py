import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

datainput = pd.read_csv("dicision.csv", delimiter=",")

X = datainput[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Data Preprocessing
from sklearn import preprocessing

label_gender = preprocessing.LabelEncoder()
label_gender.fit(['F', 'M'])
X[:, 1] = label_gender.transform(X[:, 1])

label_BP = preprocessing.LabelEncoder()
label_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = label_BP.transform(X[:, 2])

label_Chol = preprocessing.LabelEncoder()
label_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = label_Chol.transform(X[:, 3])

y = datainput["Drug"]

# train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

drugTree.fit(X_train, y_train)
predicted = drugTree.predict(X_test)

print(predicted)

print("\nDecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predicted))