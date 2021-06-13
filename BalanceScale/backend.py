#!/usr/bin/env python
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
# import csv
# import glob
# import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns; sns.set()
# import subprocess
import sys
import time
import warnings

from joblib import dump, load
from math import ceil
from matplotlib.ticker import FormatStrFormatter
from multiprocessing import cpu_count
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from tqdm import tqdm

np.random.seed(0)
warnings.filterwarnings('ignore')


# In[8]:


class CLF:
    """
    Class for classifying texts using Support Vector Machines
    
    ...
    Attributtes
    -----------
    scores_files : [str]
        Calculated scores
    out_file : str
        Output file (with desired extension e.g. .png, .pdf, .svg)
    scaler : scaler, optionals
        Scaler object
    random_state : int, optional
        Seed to use for random number generator.
    """

    def __init__(self, X, labels, names, out_file, scaler=StandardScaler(), random_state=0, classifier=GaussianNB(),
            n_classes=3, sel_labels=None, target_names=None, verbose=False, stratify=True):
        """
        Class for classifying texts using RandomForestTrees
        """

        self.X = X
        self.labels = labels
        self.names = names
        self.out_file = out_file
        self.metrics = []
        self.verbose = verbose
        self.scaler = scaler
        self.C = 0.1
        self.random_state = random_state
        self.target_names = target_names
        self.sel_labels = sel_labels
        self.accuracy = None
        self.f1 = None
        self.kappa = None
        self.auc = None
        self.precision = None
        self.recall = None
        self.stratify = stratify
        self.classifier = classifier
        self.n_classes = n_classes
        
    def _clf(self):
        """Support Vector Machine Learning"""

        
        out_split = self.out_file.split('.')
        out_f = out_split[0]
        out_ext = out_split[-1]

#         self.scores[np.isinf(self.scores)] = np.nan
        imp = SimpleImputer(missing_values=np.nan, strategy='constant')
        imp = imp.fit(self.X)
        data = self.scaler.fit_transform(imp.transform(self.X))
#         print(f'Data means: {np.mean(data, axis=0)}, stds: {np.std(data, axis=0)}')

        target = self.labels
        
        if not self.target_names is None:
            target_names = self.target_names#['random', 'non-random']
        else:
            target_names =  []
            [target_names.append(n) for n in self.labels if n not in target_names]

        idxs = np.arange(len(target))
        if self.stratify:
            self.stratify = target

        X_train, X_test, y_train, y_test, idxtrain, idxtest = train_test_split(data, target, idxs,
            random_state=self.random_state, stratify=self.stratify)

        model = self.classifier.fit(X_train, y_train)
        yfit = model.predict(X_test)
        mat = confusion_matrix(y_test, yfit)
        self.f1 = fbeta_score(y_test, yfit, 1, average='macro')
        self.accuracy = accuracy_score(y_test, yfit)       
        self.precision = precision_score(y_test, yfit, average='macro')
        self.recall = recall_score(y_test, yfit, average='macro')
        
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
            
        fpr, tpr, _ = roc_curve(y_test, yfit, pos_label=y_test[0])
        roc_auc = auc(fpr, tpr)    
        self.auc = roc_auc
        
        if not target is None:
            target_names = np.unique(target)
        else:
            target_names =  []
            [target_names.append(n) for n in self.labels if n not in target_names]

        report = classification_report(y_test, yfit,
                                target_names=target_names, output_dict=True)

        
        
        df = pd.DataFrame(report).transpose()
        mat = confusion_matrix(y_test, yfit)

        sns.set(font_scale=3)
        f, ax = plt.subplots(figsize=(20, 15))

        rc={'font.size': 20, 'axes.labelsize': 20, 'legend.fontsize': 20, 
            'axes.titlesize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
        
        sns.heatmap(mat, ax=ax,square=True, annot=True, annot_kws={"size": 160},
                fmt='d', cbar=False, xticklabels=target_names,
                    yticklabels=target_names, cmap=plt.cm.Greens)
        plt.xlabel('predicted label', size=60);
        plt.ylabel('true label', size=60);
        #plt.savefig(os.path.join('results', out_f + '_nb.' + out_ext), dpi=300)
        plt.close()
                      

    def run(self):
        """Analyze data using svm and/or clustering."""
        if self.verbose:
            print('Starting svm classifier...')
            print(f'Processing scores files: {self.scores_files}')
        self._clf()
        return {'f1': self.f1, 'auc': self.kappa, 'accuracy': self.accuracy, 
                'precision': self.precision, 'recall': self.recall,
               'auc': self.auc}

class Manager:
    def __init__(self):
        self.names = ['class', 'Left-Weight', 'Left-Distance',
                'Right-Weight', 'Right-Distance']
        self.classes = {0: 'L', 1: 'B', 2: 'R'}
        self.classes_inv = {v: k for k, v in self.classes.items()}

        self.df = self.get_data()

        self.df['class'] = self.df['class'].map(self.classes_inv)

    def plot_counts(self, df):
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.countplot(ax=ax, x='class', data=df)
        #labels = ax.get_xticklabels()
       # print(labels)
        #labels = [self.classes[int(l[-1])] for l in labels]
        labels = [self.classes[int(item.get_text())] for item in ax.get_xticklabels()]
        print(labels)
        ax.set_xticklabels(labels, fontsize=25)

        return fig


    def get_data(self):
        df = pd.read_csv('BalanceScale/data/balance-scale.data',  header=None, names=self.names)
        
        return df

    def random_unbalanced(self, classifier, df, names):
        X = df[self.names[1:]]
        y = df['class']

        model = CLF(X.values, y, names, 'original.svg', n_classes=3, classifier=classifier)
        results = model.run()
        print(results)
        print()
        df_results = pd.DataFrame(results, index=['unbalanced'])
    #     df_results.head()


        return df_results

    def random_sampling(self, classifier, df, df_results):
        dfrs = pd.concat([df[df['class']==self.classes_inv['L']].sample(n=49, random_state=0), 
                          df[df['class']==self.classes_inv['R']].sample(n=49, random_state=0),
                          df[df['class']==self.classes_inv['B']].sample(n=49, random_state=0)])
        #X = dfrs[self.names]
        X = dfrs[self.names[1:]]
        y = dfrs['class']
        print(X)
        print(y)

        model = CLF(X.values, y, self.names, 'random.png', classifier=classifier)

        results = model.run()
        print(results)
        print()
        df_tmp = pd.DataFrame(results, index=['random'])
        df_results = pd.concat([df_results, df_tmp])
        return df_results

    # K-means++ and Centroids

    def centroids(self, classifier, df, df_results):
        n_clusters = min(df['class'].value_counts())
        R = df[df['class']==self.classes_inv['R']][self.names]
        L = df[df['class']==self.classes_inv['L']][self.names]
        df_b = df[df['class']==self.classes_inv['B']]

        Ru = KMeans(n_clusters=n_clusters, random_state=0).fit(R)
        Lu = KMeans(n_clusters=n_clusters, random_state=0).fit(L)

        df_r = pd.DataFrame(Ru.cluster_centers_)
        df_r.columns = self.names
        df_l = pd.DataFrame(Lu.cluster_centers_)
        df_l.columns = self.names

        df_l['class'] = df_l['class'].apply(np.int64)
        df_r['class'] = df_r['class'].apply(np.int64)
        df_b['class'] = df_b['class'].apply(np.int64)

        dfu = pd.concat([df_r, df_l, df_b], axis=0)

        #X = dfu[self.names]
        X = dfu[self.names[1:]]
        y = dfu['class']
        model = CLF(X, y, self.names, 'km.svg', classifier=classifier)

        results = model.run()
        print(results)
        print()
        df_tmp = pd.DataFrame(results, index=['centroids'])
        df_results = pd.concat([df_results, df_tmp])


    # def centroid_nn(df_results):
        ##R
        values = df[df['class']==self.classes_inv['R']][self.names].values
        lengths = np.diag(np.dot(values, values.T))
        values_norm = values/lengths[:, None]
        lengths_c = np.diag(np.dot(Ru.cluster_centers_, Ru.cluster_centers_.T))
        clusters_norm = Ru.cluster_centers_/lengths_c[:, None]
        dists = np.dot(values_norm, clusters_norm.T)

        min_idxs = np.argmin(dists, axis=0)
        Ru_top = values[min_idxs]

        sorted_idxs = np.argsort(dists, axis=0)[::-1]
        selected = set()
        for i in range(sorted_idxs.shape[-1]):
            for j in range(sorted_idxs.shape[0]):
                if not sorted_idxs[j, i] in selected:
                    selected.add(sorted_idxs[j, i])
                    break
        selectedR = list(selected)

        ##L
        values = df[df['class']==self.classes_inv['L']][self.names].values
        lengths = np.diag(np.dot(values, values.T))
        values_norm = values/lengths[:, None]
        lengths_c = np.diag(np.dot(Lu.cluster_centers_, Lu.cluster_centers_.T))
        clusters_norm = Lu.cluster_centers_/lengths_c[:, None]
        dists = np.dot(values_norm, clusters_norm.T)

        min_idxs = np.argmin(dists, axis=0)
        Lu_top = values[min_idxs]

        sorted_idxs = np.argsort(dists, axis=0)[::-1]
        selected = set()
        # idxs = []
        for i in range(sorted_idxs.shape[-1]):
            for j in range(sorted_idxs.shape[0]):
                if not sorted_idxs[j, i] in selected:
                    selected.add(sorted_idxs[j, i])
                    break

        selectedL = list(selected)

        df_r = pd.DataFrame(Ru_top)
        #df_r['class'] = self.classes_inv['R']
        df_r.columns = self.names# + ['class']
        #df_r = df_r[['class'] + self.names]
        df_l = pd.DataFrame(Lu_top)
        #df_l['class'] = self.classes_inv['L']
        df_l.columns = self.names# + ['class']
        #df_l = df_l[['class'] + self.names]
        dfu_top = pd.concat([df_r, df_l, df_b])
        X = dfu_top[self.names[1:]]
        y = dfu_top['class']
        model = CLF(X, dfu_top['class'], self.names, 'km_top1.svg', classifier=classifier)

        results = model.run()
        print(results)
        print()
        df_tmp = pd.DataFrame(results, index=['centroid_nn'])
        df_results = pd.concat([df_results, df_tmp])


    #  def centroid_nn_unique(df_results):
        df_r = df.iloc[selectedR]
        # df_r['class'] = 'R'

        df_l = df.iloc[selectedL]
        dfu_top = pd.concat([df_r, df_l, df_b])
        X = dfu_top[self.names[1:]]
        y = dfu_top['class']

        model = CLF(X, dfu_top['class'], self.names, 'km_top1_unique.svg', classifier=classifier)

        results = model.run()
        print(results)
        print()
        df_tmp = pd.DataFrame(results, index=['centroid_nn_unique'])
        df_results = pd.concat([df_results, df_tmp])
        
        return df_results


    # # K-means++ and Random sampling (k=7)

    # In[11]:


    def km_random(self, classifier, df, df_results, k):
    #     k = 7
        n_clusters = k
        R = df[df['class']==self.classes_inv['R']][self.names]
        L = df[df['class']==self.classes_inv['L']][self.names]
        df_b = df[df['class']==self.classes_inv['B']]
        Ru = KMeans(n_clusters=n_clusters, random_state=0).fit(R)
        Lu = KMeans(n_clusters=n_clusters, random_state=0).fit(L)

        Rs = []
        Ls = []
        # _, idxs = np.unique(Ru.labels_, return_index=True)
        for i in range(n_clusters): 
            idxs = np.random.permutation(np.argwhere(Ru.labels_==i).ravel())[:k]
            df_sample = df[df['class']==self.classes_inv['R']].iloc[idxs]
            Rs.append(df_sample)
        for i in range(n_clusters): 
            idxs = np.random.permutation(np.argwhere(Lu.labels_==i).ravel())[:k]
            df_sample = df[df['class']==self.classes_inv['L']].iloc[idxs]
            Ls.append(df_sample)
        dfkrs = pd.concat([pd.concat(Rs), pd.concat(Ls), df[df['class']==self.classes_inv['B']]])

        X = dfkrs[self.names[1:]]
        y = dfkrs[['class']]
        X.shape, y.shape

        model = CLF(X, y, self.names, 'km_random.svg', classifier=classifier)
        results = model.run()
        print(results)
        print()
        df_tmp = pd.DataFrame(results, index=['km_random'])
        df_results = pd.concat([df_results, df_tmp])


    # def km_topN_unique(df_results):
        ##R
        values = df[df['class']==self.classes_inv['R']][self.names].values
        lengths = np.diag(np.dot(values, values.T))
        values_norm = values/lengths[:, None]
        lengths_c = np.diag(np.dot(Ru.cluster_centers_, Ru.cluster_centers_.T))
        clusters_norm = Ru.cluster_centers_/lengths_c[:, None]
        dists = np.dot(values_norm, clusters_norm.T)

        min_idxs = np.argmin(dists, axis=0)
        Ru_top = values[min_idxs]

        sorted_idxs = np.argsort(dists, axis=0)[::-1]
        lookup = set()
        selected = {}
        count = 0
        for i in range(sorted_idxs.shape[-1]):
            for j in range(sorted_idxs.shape[0]):        
                try:
                    if not sorted_idxs[j, i] in lookup:
                        selected[i].add(sorted_idxs[j, i])
                        lookup.add(sorted_idxs[j, i])
                        if len(selected[i]) >67:
                            break
                except KeyError:
                    selected[i] = {sorted_idxs[j, i]}
                    lookup.add(sorted_idxs[j, i])


        selectedR = [item for sublist in list(selected.values()) for item in sublist]

        ##L
        values = df[df['class']==self.classes_inv['L']][self.names].values
        lengths = np.diag(np.dot(values, values.T))
        values_norm = values/lengths[:, None]
        lengths_c = np.diag(np.dot(Lu.cluster_centers_, Lu.cluster_centers_.T))
        clusters_norm = Ru.cluster_centers_/lengths_c[:, None]
        dists = np.dot(values_norm, clusters_norm.T)

        min_idxs = np.argmin(dists, axis=0)
        Ru_top = values[min_idxs]

        sorted_idxs = np.argsort(dists, axis=0)[::-1]
        lookup = set()
        selected = {}
        count = 0
        for i in range(sorted_idxs.shape[-1]):
            for j in range(sorted_idxs.shape[0]):        
                try:
                    if not sorted_idxs[j, i] in lookup:
                        selected[i].append(sorted_idxs[j, i])
                        lookup.add(sorted_idxs[j, i])
                        if len(selected[i]) > 6:
                            break
                except KeyError:
                    selected[i] = [sorted_idxs[j, i]]
                    lookup.add(sorted_idxs[j, i])


        selectedL = [item for sublist in list(selected.values()) for item in sublist]

        df_r = df.iloc[selectedR]
        df_r['class'] = self.classes_inv['R']

        df_l = df.iloc[selectedL]
        dfu_top = pd.concat([df_r, df_l, df_b])
        X = dfu_top[self.names[1:]]
        y = dfu_top['class']
    #     dfu_top.head(50)
        model = CLF(X, dfu_top['class'], self.names, 'km_topN_unique.svg', classifier=classifier)

        results = model.run()
        print(results)
        print()
        df_tmp = pd.DataFrame(results, index=['km_topN_unique'])
        df_results = pd.concat([df_results, df_tmp])

        return df_results

    def plot_results(self, df_results):
        def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, x_labels=None):
            """Draws a bar plot with multiple bars per data point.

            Parameters
            ----------
            ax : matplotlib.pyplot.axis
                The axis we want to draw our plot on.

            data: dictionary
                A dictionary containing the data we want to plot. Keys are the names of the
                data, the items is a list of the values.

                Example:
                data = {
                    "x":[1,2,3],
                    "y":[1,2,3],
                    "z":[1,2,3],
                }

            colors : array-like, optional
                A list of colors which are used for the bars. If None, the colors
                will be the standard matplotlib color cyle. (default: None)

            total_width : float, optional, default: 0.8
                The width of a bar group. 0.8 means that 80% of the x-axis is covered
                by bars and 20% will be spaces between the bars.

            single_width: float, optional, default: 1
                The relative width of a single bar within a group. 1 means the bars
                will touch eachother within a group, values less than 1 will make
                these bars thinner.

            legend: bool, optional, default: True
                If this is set to true, a legend will be added to the axis.
            """

            # Check if colors where provided, otherwhise use the default color cycle
            if colors is None:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # Number of bars per group
            n_bars = len(data)

            # The width of a single bar
            bar_width = total_width / n_bars

            # List containing handles for the drawn bars, used for the legend
            bars = []

            # Iterate over all data
            for i, (name, values) in enumerate(data.items()):
                # The offset in x direction of that bar
                x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

                # Draw a bar for every value of that type
                for x, y in enumerate(values):
                    bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

                # Add a handle to the last drawn bar, which we'll need for the legend
                bars.append(bar[0])
            ax.set_xticklabels([''] + list(x_labels), fontsize=25)
            plt.yticks(fontsize=15);
            if legend:
                ax.legend(bars, data.keys(), fontsize=25)
        
        data = {df_results.T.index[i]: df_results.T.iloc[i] for i in range(len(df_results.T))}
        fig, ax = plt.subplots(figsize=(25, 10))
        bar_plot(ax, data, total_width=.8, single_width=.9, x_labels=df_results.index)

        return fig
