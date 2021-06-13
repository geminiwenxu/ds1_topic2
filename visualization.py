import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def combine_data(path):
    total_df = pd.DataFrame
    minority = pd.read_csv('Data/minority_class', delimiter=',')
    minority.drop(['Unnamed: 0'], axis=1, inplace=True)
    if path == "centroids":
        df = pd.read_csv("Data/minority_class", delimiter=',')
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


def plot_classification_report(cr, name, with_avg_total=False, cmap=plt.cm.Blues):
    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2: (len(lines) - 5)]:
        # print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(f"{name}")
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.xlabel('Measures')


def logistic_regression(df, name):
    imbalanced_df = pd.read_csv("/Data/numerical_data.csv", delimiter=',')
    cols = imbalanced_df.columns.tolist()
    training_cols = cols[:39]
    class_name = ['negative', 'positive']
    X = df[training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
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
    plot_classification_report(classification_report(y_test, y_pred, target_names=class_name), name)


if __name__ == '__main__':
    imbalanced_df = pd.read_csv("/Data/numerical_data.csv", delimiter=',')
    cols = imbalanced_df.columns.tolist()
    training_cols = cols[:39]
    ls_paths = ["imbalanced", "centroids", "random",
                "one_neig", "n_neig"]

    # total_df = combine_data(ls_paths[0], cols)
    # logistic_regression(total_df, ls_paths[0])
    # plt.savefig(f"{ls_paths[0]}.")

    # total_df = combine(ls_paths[1], cols)
    # logistic_reg(total_df, ls_paths[1])
    # plt.savefig(f"{ls_paths[1]}.")

    # total_df = combine(ls_paths[2], cols)
    # logistic_reg(total_df, ls_paths[2])
    # plt.savefig(f"{ls_paths[2]}.")

    # total_df = combine(ls_paths[3], cols)
    # logistic_reg(total_df, ls_paths[3])
    # plt.savefig(f"{ls_paths[3]}.")

    # total_df = combine(ls_paths[4], cols)
    # logistic_reg(total_df, ls_paths[4])
    # plt.savefig(f"{ls_paths[4]}.")
