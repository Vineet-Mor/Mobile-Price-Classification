import pandas as pd

data = pd.read_csv("Samples500.csv")
y = data["price_range"]
del data["price_range"]


def create_dataset1(data1):
    # combining three-g and four-g function
    data1 = pd.DataFrame(data1)
    data1 = data1.reset_index(drop=True)
    data1.columns = data.columns
    c34 = []
    for i in range(len(data1["four_g"])):
        if (data1["four_g"][i] == 1 and data1["three_g"][i] == 1):
            c34.append(2)
        elif (data1["three_g"][i] == 1):
            c34.append(1)
        else:
            c34.append(0)
    del data1["four_g"]
    del data1["three_g"]
    data1["c34"] = c34

    del data1["fc"]
    del data1["touch_screen"]
    del data1["mobile_wt"]
    del data1["clock_speed"]
    return data1


def create_dataset2(data1):
    data1 = np.delete(data1, [1, 8, 15, 17, 14, 2, 19, 18, 9, 6, 5, 4, 10, 3, 7], axis=1)
    return data1


def max(arr):
    a = 0
    k = 0
    b = -1
    for i in arr:
        b = b + 1
        if i > a:
            a = i
            k = b
    return a, k, b


def accuracy(y_test, y_pred):
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred) * 100)
    print("")
    print("")


def confussion_matrix_classification_report(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(y_test, y_pred))

    print("")
    print("")
    print(classification_report(y_test, y_pred))
    print("")
    print("")


def rmse(y_test, y_pred):
    import math
    MeanSquareError = np.square(np.subtract(y_test, y_pred)).mean()
    RootMeanSquareError = math.sqrt(MeanSquareError)
    print("Root Mean Square Error:\n")
    print(RootMeanSquareError)
    print("")
    print("")


def random_forest(train, test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=19, random_state=0)
    clf.fit(train, y_train)
    y_pred = clf.predict(test)
    accuracy(y_test, y_pred)
    confussion_matrix_classification_report(y_test, y_pred)
    rmse(y_test, y_pred)


def svc_model(train, test, y_train, y_test):
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(train, y_t)
    y_pred = svc.predict(test)
    accuracy(y_test, y_pred)
    confussion_matrix_classification_report(y_test, y_pred)
    rmse(y_test, y_pred)


from sklearn import model_selection as ms

train_data, test_data, y_t, y_test = ms.train_test_split(data, y, test_size=0.25, random_state=0);
train = train_data
test = test_data

# Dataset_1_train=create_dataset1(train)
# Dataset_1_test=create_dataset1(test)
# from sklearn.preprocessing import StandardScaler
# scale=StandardScaler()
# train=scale.fit_transform(Dataset_1_train)
# test=scale.transform(Dataset_1_test)

from sklearn.preprocessing import StandardScaler

scale1 = StandardScaler()
train = scale1.fit_transform(train)
test = scale1.transform(test)
import numpy as np

Dataset_1_train = create_dataset1(train)
Dataset_1_test = create_dataset1(test)
# train=train_data
# test=test_data

Dataset_2_train = create_dataset2(train)
Dataset_2_test = create_dataset2(test)

for i in range(2):
    for j in range(2):
        if j == 0 and i == 0:
            print("Random_Forest_Classifier with Dataset1 -------------------------------------")
            print("")
            random_forest(Dataset_1_train, Dataset_1_test, y_t, y_test)
        elif j == 0 and i == 1:
            print("")
            print("Random_Forest_Classifier with Dataset2 -------------------------------------")
            print("")
            random_forest(Dataset_2_train, Dataset_2_test, y_t, y_test)
        elif j == 1 and i == 0:
            print("")
            print("Support Vector Classifier with Dataset1 ------------------------------------")
            print("")
            svc_model(Dataset_1_train, Dataset_1_test, y_t, y_test)
        else:
            print("")
            print("Support Vector Classifier with Dataset1 ------------------------------------")
            print("")
            svc_model(Dataset_2_train, Dataset_2_test, y_t, y_test)


