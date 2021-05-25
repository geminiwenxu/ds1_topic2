import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def combine(path, cols):
    total_df = pd.DataFrame
    minority = pd.read_csv('Data/minority_class', delimiter=',')
    minority.drop(['Unnamed: 0'], axis=1, inplace=True)
    if path == "centroids":
        df = pd.read_csv("Data/centroids.csv", delimiter=',')
        df.columns = cols
        total_df = pd.concat([df, minority])
    elif path == "random":
        df = pd.read_csv('Data/random.csv', delimiter=',')
        df.drop(['Unnamed: 0', 'cluster'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "one_neig":
        df = pd.read_csv('Data/one_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "n_neig":
        df = pd.read_csv('Data/n_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    else:
        df = pd.read_csv("Data/numerical_data.csv", delimiter=',')
        total_df = df
    print(total_df)
    return total_df


def logistic_reg(df):
    class_name = ['negative', 'positive']
    X = df[training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    ls_accuracy = []
    ls_precision = []
    ls_recall = []
    ls_f1 = []
    ls_auc = []
    for num_iter in np.linspace(10, 100, 90):
        logistic_regression = LogisticRegression(max_iter=num_iter)
        logistic_regression.fit(X_train, y_train)
        y_pred = logistic_regression.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        ls_accuracy.append(accuracy)
        precision = metrics.precision_score(y_test, y_pred)
        ls_precision.append(precision)
        recall = metrics.recall_score(y_test, y_pred)
        ls_recall.append(recall)
        f1 = metrics.f1_score(y_test, y_pred)
        ls_f1.append(f1)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
        auc = metrics.auc(fpr, tpr)
        ls_auc.append(auc)
    logistic_regression = LogisticRegression(max_iter=100)
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    print("after 100 iterations: ", accuracy, precision, recall, f1, auc)
    print(classification_report(y_test, y_pred, target_names=class_name))
    return ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc


if __name__ == '__main__':
    imbalanced_df = pd.read_csv("Data/numerical_data.csv", delimiter=',')
    cols = imbalanced_df.columns.tolist()
    training_cols = cols[:39]

    ls_paths = ["imbalanced", "centroids", "random",
                "one_neig", "n_neig"]
    for name in ls_paths:
        total_df = combine(name, cols)
        print(total_df.shape)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
        plt.plot(np.linspace(10, 100, 90), ls_accuracy, label=f"{name}")
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()
    plt.show()
