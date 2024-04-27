import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Styling Code  
# Sidebar section for dataset selection
st.sidebar.title("Choose Dataset")
dataset_name = st.sidebar.radio("Select Dataset", ("IRIS", "Digits"))


# Adding CSS For Header & Predicted Class
st.write(
    f"""
    <style>
    h1 {{
        text-align: center;
        color: #00FFFF;
    }}
    .predicted {{
        font-weight: bold;
        color: green;
    }}
    </style>
    <h1>Mukesh ML Classifier using Streamlit</h1>
    """,
    unsafe_allow_html=True
)


# Load datasets
iris = datasets.load_iris()
digits = datasets.load_digits()


# Defining classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}


# Loading Datasets using radio button
if dataset_name == "IRIS":
    data = iris
else:
    data = digits


# Actual Machine Learning Starts here
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)


# Choosing Classifier
classifier_name = st.sidebar.selectbox("Choose Classifier", list(classifiers.keys()))
classifier = classifiers[classifier_name]
classifier.fit(X_train, y_train)

# Choosing Input Features using Horizontal Bars
st.sidebar.subheader("Input Feature Values")
feature_values = []
for i in range(len(data.feature_names)):
    min_val = float(data.data[:, i].min())
    max_val = float(data.data[:, i].max())
    if min_val == max_val:
        min_val -= 0.1
        max_val += 0.1
    feature_values.append(st.sidebar.slider(data.feature_names[i], min_val, max_val))

# Class Prediction along with predict button
if st.sidebar.button("Predict", help="predict-button", key="predict-button"):
    prediction = classifier.predict([feature_values])
    st.write(f"<span class='predicted'>Predicted Class: {prediction[0]}</span>", unsafe_allow_html=True)
