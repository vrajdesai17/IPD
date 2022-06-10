import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

def app():
    st.title('Machine Learning')

    st.write("""
    # Explore different classifier and datasets
    Which one is the best?
    """)

    dataset_name = st.sidebar.selectbox(
        'Select Dataset',
        ('Iris', 'Breast Cancer', 'Wine')
    )

    st.write(f"## {dataset_name} Dataset")

    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('KNN', 'SVM', 'Random Forest')
    )

    def get_dataset(name):
        data = None
        if name == 'Iris':
            data = datasets.load_iris()
        elif name == 'Wine':
            data = datasets.load_wine()
        else:
            data = datasets.load_breast_cancer()
        X = data.data
        y = data.target #  The distinguishing feature of the target array is that it is usually the quantity we want to predict from the data: in statistical terms, it is the dependent variable. For example, in the preceding data we may wish to construct a model that can predict the species of flower based on the other measurements; in this case, the species column would be considered the target array.
        return X, y

    X, y = get_dataset(dataset_name)
    st.write('Shape of dataset:', X.shape)
    st.write('number of classes:', len(np.unique(y)))

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0) # C parameter in SVM is Penalty parameter of the error term. You can consider it as the degree of correct classification that the algorithm has to meet or the degree of optimization the the SVM has to meet. 
            params['C'] = C
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15) # nearest neighbours
            params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100) #  This is the number of trees you want to build before taking the maximum voting or averages of predictions. 
            params['n_estimators'] = n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        else:
            clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=1234) # random_state simply sets a seed to the random generator, so that your train-test splits are always deterministic. If you don't set a seed, it is different each time.
        return clf

    clf = get_classifier(classifier_name, params)
    #### CLASSIFICATION ####

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) # train_test_split selects randomly the train and test size basing on the ratio given.

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2) # Principal Component Analysis is basically a statistical procedure to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables
    X_projected = pca.fit_transform(X)
    # .fit() : perform the calculation on the feature values of input data and fit this calculation to the transformer.
    # .transform(): For changing the data we probably do transform, in the transform() method, where we apply the calculations that we have calculated in fit() to every data point in feature F. 
    # .fit_transform() : combination of fit and transform.
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure() # The plt is a common alias of matplotlib.
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')
    # alpha is the transparency of fig
    # cmap is color map in matplotlib

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    #plt.show()
    st.pyplot(fig)
