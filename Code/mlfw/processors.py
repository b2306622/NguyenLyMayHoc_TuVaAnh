# -*- coding: utf-8 -*-
import os.path
import sys
from time import time

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from helpers import Files, TextFeaturesHelper, SiftFeatureHelper, OtherFeaturesHelper, FeaturesMerger
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing

sys.modules['sklearn.externals.joblib'] = joblib
import pickle


class Data:
    def __init__(self, training_path, testing_path, n_classes):
        # importlib.reload(sys)  # had to deal with 'unicode' issues :/
        # sys.setdefaultencoding('utf8')
        self.n_classes = n_classes
        self.training_files = Files(training_path, n_classes)
        self.testing_files = Files(testing_path, n_classes)
        self.training_text_file_ids = []
        self.training_text_texts = []
        self.training_text_classes = []
        self.training_text_tfidf = []
        self.testing_text_file_ids = []
        self.testing_text_texts = []
        self.testing_text_classes = []
        self.testing_text_tfidf = []
        self.testing_text_predicted = []
        self.testing_text_predicted_prob = []
        self.text_pre = None
        
        training_text_objects_file = os.path.join(training_path,'training_text_objects.pkl')
        testing_text_objects_file = os.path.join(testing_path,'testing_text_objects.pkl')
        if os.path.exists(training_text_objects_file) and os.path.exists(testing_text_objects_file):
            with open(training_text_objects_file, 'rb') as f:
                self.training_text_file_ids, self.training_text_texts, self.training_text_classes, self.training_text_tfidf = pickle.load(f)
                f.close()
            with open(testing_text_objects_file, 'rb') as f:
                self.testing_text_file_ids, self.testing_text_texts, self.testing_text_classes, self.testing_text_tfidf = pickle.load(f)
                f.close()
        else:
            self.prepare_text_data()
            self.do_build_text_tfidf_transformer()
            self.do_training_tfidf_estimate()
            self.do_testing_tfidf_estimate()
            with open(training_text_objects_file, 'wb') as f:
                pickle.dump([ self.training_text_file_ids, self.training_text_texts, self.training_text_classes, self.training_text_tfidf ],f)
                f.close()
            with open(testing_text_objects_file, 'wb') as f:
                pickle.dump([ self.testing_text_file_ids, self.testing_text_texts, self.testing_text_classes, self.testing_text_tfidf ], f)
                f.close()

        self.training_image_file_ids = []
        self.training_image_classes = []
        self.training_image_tfidf = []
        self.testing_image_file_ids = []
        self.testing_image_classes = []
        self.testing_image_tfidf = []
        self.testing_image_predicted = []
        self.testing_image_predicted_prob = []

        training_image_objects_file = os.path.join(training_path,'training_image_objects.pkl')
        testing_image_objects_file = os.path.join(testing_path,'testing_image_objects.pkl')
        if os.path.exists(training_image_objects_file) and os.path.exists(testing_image_objects_file):
            with open(training_image_objects_file, 'rb') as f:
                self.training_image_file_ids, self.training_image_classes, self.training_image_tfidf = pickle.load(f)
                f.close()
            with open(testing_image_objects_file, 'rb') as f:
                self.testing_image_file_ids, self.testing_image_classes, self.testing_image_tfidf = pickle.load(f)
                f.close()
        else:
            self.prepare_image_data()
            with open(training_image_objects_file, 'wb') as f:
                pickle.dump([ self.training_image_file_ids, self.training_image_classes, self.training_image_tfidf ],f)
                f.close()
            with open(testing_image_objects_file, 'wb') as f:
                pickle.dump([ self.testing_image_file_ids, self.testing_image_classes, self.testing_image_tfidf ], f)
                f.close()


    def prepare_text_data(self):
        text_helper = TextFeaturesHelper()
        text_helper.load_texts(self.training_files)
        self.training_text_texts = text_helper.texts
        self.training_text_classes = text_helper.classes
        self.training_text_file_ids = text_helper.file_ids
        self.training_text_tfidf = []
        text_helper.load_texts(self.testing_files)
        self.testing_text_file_ids = text_helper.file_ids
        self.testing_text_texts = text_helper.texts
        self.testing_text_classes = text_helper.classes
        self.testing_text_tfidf = []
        self.text_pre = None

    def do_build_text_tfidf_transformer(self):
        print("Building text tf-idf transformer ... ")
        self.text_pre = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ])
        self.text_pre.fit(self.training_text_texts)
        print("Done")

    def do_training_tfidf_estimate(self):
        self.training_text_tfidf = self.text_pre.transform(self.training_text_texts)

    def do_testing_tfidf_estimate(self):
        self.testing_text_tfidf = self.text_pre.transform(self.testing_text_texts)

    def prepare_image_data(self):
        image_helper = SiftFeatureHelper(img_width=300, n_clusters=1000)
        image_helper_other = OtherFeaturesHelper(img_width=300, color=False, hog=False, gist=True, deep=False)
        print("Building image BOW vocab ... ")
        t0 = time()
        path = self.training_files.dataset_path + '/' + 'voc.pkl'
        if os.path.exists(path):
            voc = joblib.load(path)
            image_helper.set_voc(voc)
        else:
            image_helper.develop_vocabulary(self.training_files)
            joblib.dump(image_helper.voc, path)
        print("done in %0.3fs" % (time() - t0))

        print("Building training image features ... ")
        t0 = time()
        image_helper.build_BOW_features_classes(self.training_files)
        image_helper_other.build_features_classes(self.training_files)
        training_sift_classes = image_helper.classes
        training_sift_features = image_helper.features
        training_sift_file_ids = image_helper.file_ids
        training_other_classes = image_helper_other.classes
        training_other_features = image_helper_other.features
        training_other_file_ids = image_helper_other.file_ids
        training_merger = FeaturesMerger(training_sift_file_ids, training_sift_features, training_sift_classes,
                                         training_other_file_ids, training_other_features, training_other_classes)
        # training_merger = FeaturesMerger(training_sift_file_ids, training_sift_features, training_sift_classes,[],
        # [], []) training_merger = FeaturesMerger([], [], [],training_other_file_ids, training_other_features,
        # training_other_classes)
        print("Done in %0.3fs" % (time() - t0))
        self.training_image_classes = training_merger.classes
        self.training_image_file_ids = training_merger.paths
        self.training_image_tfidf = training_merger.features

        print("Building testing image features ... ")
        t0 = time()
        image_helper.build_BOW_features_classes(self.testing_files)
        image_helper_other.build_features_classes(self.testing_files)
        testing_sift_classes = image_helper.classes
        testing_sift_features = image_helper.features
        testing_sift_file_ids = image_helper.file_ids
        testing_other_classes = image_helper_other.classes
        testing_other_features = image_helper_other.features
        testing_other_file_ids = image_helper_other.file_ids
        testing_merger = FeaturesMerger(testing_sift_file_ids, testing_sift_features, testing_sift_classes,
                                        testing_other_file_ids, testing_other_features, testing_other_classes)
        # testing_merger = FeaturesMerger(testing_sift_file_ids, testing_sift_features, testing_sift_classes,[], [],
        # []) testing_merger = FeaturesMerger([], [], [],testing_other_file_ids, testing_other_features,
        # testing_other_classes)
        self.testing_image_file_ids = testing_merger.paths
        self.testing_image_classes = testing_merger.classes
        self.testing_image_tfidf = testing_merger.features
        print("Done in %0.3fs" % (time() - t0))


class Text:
    def __init__(self, data):
        self.data = data

    def do_test_clasifiers(self):

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes"]

        # names = ["Nearest Neighbors", "Linear SVM",
        #         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #         "Naive Bayes"]

        # param_grid = {'C': [0.1, 0.5, 1, 5, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
        #                   'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5], }

        param_grid = {'C': [0.025, 0.1, 1, 5, 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.01, 1.0, 1.5], }
        classifiers = [
            KNeighborsClassifier(3),
            # SVC(kernel="linear", C=0.025),
            GridSearchCV(SVC(kernel='linear', class_weight='balanced', probability=True), param_grid, cv=3),
            GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid, cv=3),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            MultinomialNB()]

        x_train = self.data.training_text_tfidf
        y_train = self.data.training_text_classes
        x_test = self.data.testing_text_tfidf
        y_test = self.data.testing_text_classes

        for name, clf in zip(names, classifiers):
            score = 0
            t0 = time()
            clf.fit(x_train, y_train)
            if name == "RBF SVM":
                score = clf.best_estimator_.score(x_test, y_test)
            else:
                score = clf.score(x_test, y_test)
            print(name + ":" + str(score) + ":" + str(time() - t0))

    def do_test_clasifier(self, name):

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes"]

        param_grid = {'C': [0.025, 0.1, 1, 5, 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.01, 1.0, 1.5], }
        classifiers = [
            KNeighborsClassifier(3),
            # SVC(kernel="linear", C=0.025),
            GridSearchCV(SVC(kernel='linear', class_weight='balanced', probability=True), param_grid, cv=3),
            GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid, cv=3),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            MultinomialNB()]

        x_train = self.data.training_text_tfidf
        y_train = self.data.training_text_classes
        x_test = self.data.testing_text_tfidf
        y_test = self.data.testing_text_classes

        index  = names.index(name)
        if index >=0 and index < len(names):
            t0 = time()
            print("Training " + name + " classifier for text\n")
            clf = classifiers[index]
            clf.fit(x_train, y_train)
            print("Testing " + name + " classifier for text\n")
            if name == "RBF SVM":
                scores = clf.best_estimator_.predict_proba(x_test)
            else:
                scores = clf.predict_proba(x_test)
            print("Done in : " + str(time() - t0)+"\n\n")
            self.data.testing_text_predicted_prob = scores

    @staticmethod
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def do_test_clustering(self):
        # Number of clusters
        n_clusters = 4
        # Distance metric: 'cosine', 'euclidean', 'cityblock'
        metric = "euclidean"
        # Linkage: 'ward', 'complete', 'average', 'single'
        linkage = "ward"
        names = ["HAC", "kMeans", "Spectral"]
        models = [
            AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=metric),
            MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1000, init_size=1000, batch_size=1000),
            SpectralClustering(n_clusters=n_clusters, n_init=1000)]

        x_train = self.data.training_text_tfidf.toarray()
        print("Number of text features: %d " % len(x_train[0]))
        y_train = self.data.training_text_classes
        x_test = self.data.testing_text_tfidf
        y_test = self.data.testing_text_classes
        print(np.array(y_train).flatten() - 1)
        for name, model in zip(names, models):
            t0 = time()
            print("Performing %s clustering ... " % name)
            model.fit_predict(x_train)
            print("Done in %0.3fs" % (time() - t0))
            print(model.labels_)
            #if name == "HAC":
            #    plt.title('Hierarchical Clustering Dendrogram')
                # plot the top three levels of the dendrogram
            #    self.plot_dendrogram(model, truncate_mode='level', p=10)
            #    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            #    plt.show()
            print("Homogeneity: %0.3f" % metrics.homogeneity_score(np.array(y_train).flatten() - 1, model.labels_))
            print("Completeness: %0.3f" % metrics.completeness_score(np.array(y_train).flatten() - 1, model.labels_))
            print("V-measure: %0.3f" % metrics.v_measure_score(np.array(y_train).flatten() - 1, model.labels_))
            print()


class Image:
    def __init__(self, data):
        self.data = data

    def do_test_clasifiers(self):

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes"]

        # names = ["Nearest Neighbors", "Linear SVM",
        #         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #         "Naive Bayes"]

        param_grid = {'C': [0.025, 0.1, 1, 5],  # , 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.01, 1.0, 1.5], }
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, probability=True),
            GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid, cv=3),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB()]

        x_train = self.data.training_image_tfidf
        y_train = self.data.training_image_classes
        x_test = self.data.testing_image_tfidf
        y_test = self.data.testing_image_classes

        for name, clf in zip(names, classifiers):
            score = 0
            pred_image = []
            t0 = time()
            clf.fit(x_train, y_train)
            if name == "RBF SVM":
                score = clf.best_estimator_.score(x_test, y_test)
                pred_image = clf.best_estimator_.predict_proba(x_test)
            else:
                score = clf.score(x_test, y_test)
                pred_image = clf.predict_proba(x_test)
            print(name + ":" + str(score) + ":" + str(time() - t0))

    def do_test_clasifier(self, name):

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes"]

        # names = ["Nearest Neighbors", "Linear SVM",
        #         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #         "Naive Bayes"]

        param_grid = {'C': [0.025, 0.1, 1, 5],  # , 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.01, 1.0, 1.5], }
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, probability=True),
            GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid, cv=3),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB()]

        x_train = self.data.training_image_tfidf
        y_train = self.data.training_image_classes
        x_test = self.data.testing_image_tfidf
        y_test = self.data.testing_image_classes

        index  = names.index(name)
        if index >=0 and index < len(names):
            t0 = time()
            print("Training " + name + " classifier for image\n")
            clf = classifiers[index]
            clf.fit(x_train, y_train)
            print("Testing " + name + " classifier for image\n")
            if name == "RBF SVM":
                scores = clf.best_estimator_.predict_proba(x_test)
            else:
                scores = clf.predict_proba(x_test)
            print("Done in : " + str(time() - t0)+"\n\n")
            self.data.testing_image_predicted_prob = scores


    @staticmethod
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def do_test_clustering(self):
        # Number of clusters
        n_clusters = 4
        # Distance metric: 'cosine', 'euclidean', 'cityblock'
        metric = "euclidean"
        # Linkage: 'ward', 'complete', 'average', 'single'
        linkage = "complete"
        names = ["HAC", "kMeans", "Spectral"]
        models = [
            AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=metric),
            MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1000, init_size=1000, batch_size=1000),
            SpectralClustering(n_clusters=n_clusters, n_init=1000)]

        x_train = self.data.training_image_tfidf
        print("Size of X: %d " % x_train[0].size)
        y_train = self.data.training_image_classes
        x_test = self.data.testing_image_tfidf
        y_test = self.data.testing_image_classes
        print(np.array(y_train).flatten() - 1)
        for name, model in zip(names, models):
            t0 = time()
            print("Performing %s clustering ... " % name)
            model.fit_predict(x_train)
            print("Done in %0.3fs" % (time() - t0))
            print(model.labels_)
            #if name == "HAC":
            #    plt.title('Hierarchical Clustering Dendrogram')
                # plot the top three levels of the dendrogram
            #    self.plot_dendrogram(model, truncate_mode='level', p=10)
            #    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            #    plt.show()
            print("Homogeneity: %0.3f" % metrics.homogeneity_score(np.array(y_train).flatten() - 1, model.labels_))
            print("Completeness: %0.3f" % metrics.completeness_score(np.array(y_train).flatten() - 1, model.labels_))
            print("V-measure: %0.3f" % metrics.v_measure_score(np.array(y_train).flatten() - 1, model.labels_))
            print()


class Combined_Classifier:
    def __init__(self, data, text, image):
        self.data = data
        self.data_frame = None
        self.prepare_data()
    
    def prepare_data(self):
        tic = time()
        embeddings = []
        print("Indexing search data...")
        for file_id in self.data.testing_text_file_ids:
            print(file_id)
            embedding = []
            embedding.append(file_id)
            embedding.append(100000)
            embedding.append(100000)
            embedding.append(100000)
            embedding.append(100000)
            embedding.append(100000)
            embedding.append(100000)
            embedding.append(100000)
            embeddings.append(embedding)
        self.data_frame = pd.DataFrame(embeddings, 
                            columns = ['file_id',
                            'text_class0', 'text_class1', 
                            'image_class0', 'image_class1', 
                            'combined_class0', 'combined_class1', 
                            'final_class'])
        toc = time()
        print("Indexed files for given data set in ", toc-tic," seconds")
    
    def get_text_ranking(self):
        test_file_ids = self.data.testing_text_file_ids
        dists = self.data.testing_text_predicted_prob
        #print(dists)
        for i in range(0, len(test_file_ids)):
            #print("%s %f %f\n" %(test_file_ids[i], dists[i][0], dists[i][1]))
            self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'text_class0'] = dists[i][0]
            self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'text_class1'] = dists[i][1]
            #print("%s %f %f\n" %(test_file_ids[i], 
            #    self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'text_class0'].values[0],
            #    self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'text_class1'].values[0] )) 

    def get_image_ranking(self):
        test_file_ids = self.data.testing_image_file_ids
        dists = self.data.testing_image_predicted_prob
        #print(dists)
        for i in range(0, len(test_file_ids)):
            #print("%s %f %f\n" %(test_file_ids[i], dists[i][0], dists[i][1]))
            self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'image_class0'] = dists[i][0]
            self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'image_class1'] = dists[i][1]

            #print("%s %f %f\n" %(test_file_ids[i], 
            #    self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'image_class0'].values[0],
            #    self.data_frame.loc[self.data_frame['file_id']==test_file_ids[i], 'image_class1'].values[0] )) 
    
    def do_combined_ranking(self, alpha):
        self.get_text_ranking()
        self.get_image_ranking()
        for index, row in self.data_frame.iterrows():
            self.data_frame.loc[index,'combined_class0'] = alpha*self.data_frame.at[index,'text_class0'] + (1.0-alpha)*self.data_frame.at[index,'image_class0']
            self.data_frame.loc[index,'combined_class1'] = alpha*self.data_frame.at[index,'text_class1'] + (1.0-alpha)*self.data_frame.at[index,'image_class1']
            if self.data_frame.at[index,'combined_class0'] > self.data_frame.at[index,'combined_class1']:
                self.data_frame.loc[index,'final_class'] = 1
            else:
                self.data_frame.loc[index,'final_class'] = 2
        print(self.data_frame)
        #print(self.data.testing_image_classes)
        print("Classification accuracy = %f" %accuracy_score(self.data.testing_image_classes, self.data_frame[["final_class"]].to_numpy()))

    def show_classification_result(self, to_file):
        page = html_page_head + html_page_body_begin
        
        page = page + '<h1>Class 1</h1><br>\n'
        page = page + '<table>\n'
        for index, row in self.data_frame.iterrows():
            if row['final_class'] == 1:
                page = page + '<tr>\n'
                image_file, text_file = self.data.testing_files.get_file_path(row['file_id'])
                #print(text_file)
                #print(str(row['text_dist']))
                page = page + '<td><embed src=\"'+text_file+'\"></td>\n' 
                page = page + '<td><img src=\"'+image_file+'\"></td>\n'
                page = page + '</tr>\n'
                page = page + '<tr>\n'
                page = page + '<td>Class 1 prob = \"'+str(row['combined_class0'])+'\"></td>\n' 
                page = page + '<td>Class 2 prob = \"'+str(row['combined_class1'])+'\"></td>\n'
                page = page + '</tr>\n'
        page = page + '</table>\n'
        page = page + '<h1>Class 2</h1><br>\n'
        page = page + '<table>\n'
        for index, row in self.data_frame.iterrows():
            if row['final_class'] == 2:
                page = page + '<tr>\n'
                image_file, text_file = self.data.testing_files.get_file_path(row['file_id'])
                #print(text_file)
                #print(str(row['text_dist']))
                page = page + '<td><embed src=\"'+text_file+'\"></td>\n' 
                page = page + '<td><img src=\"'+image_file+'\"></td>\n'
                page = page + '</tr>\n'
                page = page + '<tr>\n'
                page = page + '<td>Class 1 prob = \"'+str(row['combined_class0'])+'\"></td>\n' 
                page = page + '<td>Class 2 prob = \"'+str(row['combined_class1'])+'\"></td>\n'
                page = page + '</tr>\n'
        page = page + '</table>\n'
        page = page + html_page_body_end + html_page_end
        self.data.training_files.save_file(to_file, page)



html_page_head = '<html><head><title>Search result</title></head>\n'
html_page_body_begin = '<body>\n'
html_page_body_end = '</body>\n'
html_page_end = '</html>\n'


class Search:
    def __init__(self, data):
        self.data = data
        self.data_frame = None
        self.prepare_data()
    
    def prepare_data(self):
        tic = time()
        embeddings = []
        print("Indexing search data...")
        for file_id in self.data.training_text_file_ids:
            print(file_id)
            embedding = []
            embedding.append(file_id)
            embedding.append(100000)
            embedding.append(100000)
            embedding.append(100000)
            embeddings.append(embedding)
        self.data_frame = pd.DataFrame(embeddings, 
                            columns = ['file_id', 'text_dist', 'image_dist', 'combined_dist'])
        toc = time()
        print("Indexed files for given data set in ", toc-tic," seconds")
    
    def do_text_ranking(self, id, metric):
        train_file_ids = self.data.training_text_file_ids
        train_features = self.data.training_text_tfidf.toarray()
        test_file_ids = self.data.testing_text_file_ids
        test_features = self.data.testing_text_tfidf.toarray()
        dists = pairwise_distances(train_features, np.reshape(test_features[id], (1, -1)), metric=metric)
        dists = preprocessing.normalize(dists, norm="l1", axis=0)
        #print(dists)
        for i in range(0, len(train_file_ids)):
            print(train_file_ids[i])
            self.data_frame.loc[self.data_frame['file_id']==train_file_ids[i], 'text_dist'] = dists[i]

    def do_image_ranking(self, id, metric):
        train_file_ids = self.data.training_image_file_ids
        train_features = self.data.training_image_tfidf
        test_file_ids = self.data.testing_image_file_ids
        test_features = self.data.testing_image_tfidf
        dists = pairwise_distances(train_features, np.reshape(test_features[id], (1, -1)), metric=metric)
        dists = preprocessing.normalize(dists, norm="l1", axis=0)
        #print(dists)
        for i in range(0, len(train_file_ids)):
            self.data_frame.loc[self.data_frame['file_id']==train_file_ids[i], 'image_dist'] = dists[i]
    
    def do_combined_ranking(self, id, metric, alpha):
        self.do_text_ranking(id, metric)
        self.do_image_ranking(id, metric)
        for index, row in self.data_frame.iterrows():
            self.data_frame.loc[index,'combined_dist'] = alpha*self.data_frame.at[index,'text_dist'] + (1.0-alpha)*self.data_frame.at[index,'image_dist']
        print(self.data_frame)

    def show_top(self, k, id, medium, to_file):
        if medium == 'text':
            sorted = self.data_frame.sort_values(by='text_dist', ascending=True)
        if medium == 'image':
            sorted = self.data_frame.sort_values(by='image_dist', ascending=True)
        if medium == 'combined':
            sorted = self.data_frame.sort_values(by='combined_dist', ascending=True)
        print(sorted)
        page = html_page_head + html_page_body_begin
        page = page + '<h1>The query</h1><br>\n'
        page = page + '<table>\n'
        page = page + '<tr>\n'
        image_file, text_file  = self.data.testing_files.get_file_path(self.data.testing_image_file_ids[id])
        page = page + '<td><embed src=\"'+text_file+'\"></td>\n' 
        page = page + '<td><img src=\"'+image_file+'\"></td>\n'
        page = page + '</tr>\n'
        page = page + '</table>\n'

        page = page + '<h1>The results</h1><br>\n'
        page = page + '<table>\n'
        count = 0
        for index, row in sorted.iterrows():
            count += 1
            if count > k:
                break
            page = page + '<tr>\n'
            image_file, text_file = self.data.training_files.get_file_path(row['file_id'])
            #print(text_file)
            #print(str(row['text_dist']))
            page = page + '<td><embed src=\"'+text_file+'\"></td>\n' 
            page = page + '<td><img src=\"'+image_file+'\"></td>\n'
            page = page + '</tr>\n'
            page = page + '<tr>\n'
            page = page + '<td>Text distance=\"'+str(row['text_dist'])+'\"></td>\n' 
            page = page + '<td>Image distance=\"'+str(row['image_dist'])+'\"></td>\n'
            page = page + '<td>Combined distance=\"'+str(row['combined_dist'])+'\"></td>\n'
            page = page + '</tr>\n'
        page = page + '</table>\n'
        page = page + html_page_body_end + html_page_end
        self.data.training_files.save_file(to_file, page)
