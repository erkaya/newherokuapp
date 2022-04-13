"""Thanks to Python Engineer - Adopted from his amazing project"""
#import streamlit as st
#from tkinter import Y
import streamlit as st
from sklearn import datasets
import numpy as np




st.title("Streamlit Example")

st.write(""" 
# Explore Different Classifiers
Which one is the best?
""")


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))


classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        st.write("""
        Iris Dataset
        """)
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        st.write("""
        Breast Cancer Dataset
        """)
        data = datasets.load_breast_cancer()
    else:
        st.write("""
        Wine Dataset
        """)
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)

st.write("shape of dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        from sklearn.svm import SVC
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"])
    return clf


clf = get_classifier(classifier_name, params)

#Classification Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"classifer name: {classifier_name}")
st.write("Accuracy", acc)

#plotting the
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)
import matplotlib.pyplot as plt
X1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(X1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
#plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

