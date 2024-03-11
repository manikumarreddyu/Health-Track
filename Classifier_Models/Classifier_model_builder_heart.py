import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

url = "res/dataset/heart.csv"
heart = pd.read_csv(url)

# Ordinal feature encoding
df = heart.copy()
encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    del dummy

# Separating X and y
X = df.drop('HeartDisease', axis=1)
Y = df['HeartDisease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""________Naive Bayes Algorithm________"""
# Train the Naive Bayes classifier
nb_classifier = GaussianNB(var_smoothing=1e-9)
nb_classifier.fit(X_train, y_train)
# Predict using the Naive Bayes classifier
nb_predictions = nb_classifier.predict(X_test)
# Calculate confusion matrix and accuracy for Naive Bayes classifier
nb_cm = confusion_matrix(y_test, nb_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_classifier_report = classification_report(y_test, nb_predictions)
nb_classifier_report_dict = classification_report(y_test, nb_predictions, output_dict=True)


def plt_NB():

    def classifier_report():
        report_df = pd.DataFrame(nb_classifier_report_dict).transpose()
        # Display the classification report as a table using st.write()
        st.write("Naive Bayes Classifier Report")
        st.write(report_df)
        st.write()

    # Plot confusion matrix for Naive Bayes classifier
    plt.figure()
    plt.imshow(nb_cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix - Naive Bayes')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = nb_cm.max() / 2
    for i, j in np.ndindex(nb_cm.shape):
        plt.text(j, i, format(nb_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if nb_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    with col1:
        classifier_report()
    with col2:
        st.pyplot()



"""________KNN Algorithm________"""
# Train the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
# Predict using the K-Nearest Neighbors classifier
knn_predictions = knn_classifier.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_classifier_report = classification_report(y_test, knn_predictions)
knn_classifier_report_dict = classification_report(y_test, knn_predictions, output_dict=True)


def plt_KNN():
    def classifier_report():
        report_df = pd.DataFrame(knn_classifier_report_dict).transpose()
        # Display the classification report as a table using st.write()
        st.write("K-Nearest Neighbors Report")
        st.write(report_df)
        st.write()

    # Plot confusion matrix for Naive Bayes classifier
    plt.figure()
    plt.imshow(knn_cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix - KNN')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = knn_cm.max() / 2
    for i, j in np.ndindex(knn_cm.shape):
        plt.text(j, i, format(knn_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if knn_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    with col1:
        classifier_report()
    with col2:
        st.pyplot()



"""________Decision Tree________"""
# Train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=None)
dt_classifier.fit(X_train, y_train)
# Predict using the Decision Tree classifier
dt_predictions = dt_classifier.predict(X_test)
# Calculate confusion matrix and accuracy for Decision Tree classifier
dt_cm = confusion_matrix(y_test, dt_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_classifier_report = classification_report(y_test, dt_predictions)
dt_classifier_report_dict = classification_report(y_test, dt_predictions, output_dict=True)


def plt_DT():
    def classifier_report():
        report_df = pd.DataFrame(dt_classifier_report_dict).transpose()
        # Display the classification report as a table using st.write()
        st.write("Decision Tree Classifier Report")
        st.write(report_df)
        st.write()

    # Plot confusion matrix for Naive Bayes classifier
    plt.figure()
    plt.imshow(dt_cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix - Decision Tree')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = dt_cm.max() / 2
    for i, j in np.ndindex(dt_cm.shape):
        plt.text(j, i, format(dt_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if dt_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    with col1:
        classifier_report()
    with col2:
        st.pyplot()



"""________Logistic Regression Algorithm________"""
# Train the Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)
# Predict using the Logistic Regression classifier
lr_predictions = lr_classifier.predict(X_test)
# Calculate confusion matrix and accuracy for Logistic Regression classifier
lr_cm = confusion_matrix(y_test, lr_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_classifier_report = classification_report(y_test, lr_predictions)
lr_classifier_report_dict = classification_report(y_test, lr_predictions, output_dict=True)


def plt_LR():
    def classifier_report():
        report_df = pd.DataFrame(lr_classifier_report_dict).transpose()
        # Display the classification report as a table using st.write()
        st.write("Logistic Regression Classifier Report")
        st.write(report_df)
        st.write()

    # Plot confusion matrix for classifier
    plt.figure()
    plt.imshow(lr_cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = lr_cm.max() / 2
    for i, j in np.ndindex(lr_cm.shape):
        plt.text(j, i, format(lr_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if lr_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    with col1:
        classifier_report()
    with col2:
        st.pyplot()



"""________Random Forest Algorithm________"""
# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)
# Predict using the Random Forest classifier
rf_predictions = rf_classifier.predict(X_test)
# Calculate confusion matrix and accuracy for Random Forest classifier
rf_cm = confusion_matrix(y_test, rf_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classifier_report = classification_report(y_test, rf_predictions)
rf_classifier_report_dict = classification_report(y_test, rf_predictions, output_dict=True)


def plt_RF():
    def classifier_report():
        report_df = pd.DataFrame(rf_classifier_report_dict).transpose()
        # Display the classification report as a table using st.write()
        st.write("Random Forest Classifier Report")
        st.write(report_df)
        st.write()

    # Plot confusion matrix for  classifier
    plt.figure()
    plt.imshow(rf_cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix - Random Forest')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = rf_cm.max() / 2
    for i, j in np.ndindex(rf_cm.shape):
        plt.text(j, i, format(rf_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if rf_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    with col1:
        classifier_report()
    with col2:
        st.pyplot()


# Selecting the best suitable algorithm based on classifier_report
models = {
    'Naive Bayes': nb_classifier_report,
    'K-Nearest Neighbors (KNN)': knn_classifier_report,
    'Decision Tree': dt_classifier_report,
    'Logistic Regression': lr_classifier_report,
    'Random Forest': rf_classifier_report
}
best_model = max(models, key=models.get)

# Saving the model
pickle.dump(nb_classifier, open('res/pickle/heart_disease_classifier_NB.pkl', 'wb'))
pickle.dump(knn_classifier, open('res/pickle/heart_disease_classifier_KNN.pkl', 'wb'))
pickle.dump(dt_classifier, open('res/pickle/heart_disease_classifier_DT.pkl', 'wb'))
pickle.dump(lr_classifier, open('res/pickle/heart_disease_classifier_LR.pkl', 'wb'))
pickle.dump(rf_classifier, open('res/pickle/heart_disease_classifier_RF.pkl', 'wb'))

