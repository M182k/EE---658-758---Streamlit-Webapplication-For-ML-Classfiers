# EE 658/758 Streamlit - Web App FOR ML Classifiers - Assignment

A web application using Python and Streamlit that allows users to interact with machine learning models for 2 Datasets viz two datasets, IRIS and Digits dataset


1. The Entire Web application has python, CSS (for styling)

2. In this application we are going to Build 3 Machine Learning Classifiers and Integrating with an application using streamlit (Python framework)

3. ML Classifiers are built on 2 Datasets (IRIS, Digits) downloaded from the sci-kit learn library


4. Below are the Techstack (frameworks, libraries) we are using in this project
   For Building ML Classfiers, Datasets - Scikit learn 
                   Web application      - Streamlit
                   Styling             - CSS


# 5. Code Walkthrough / Process

## 5.1. Import & Download required libraries 

   a. **Import & Download required libraries**:
      
   b. **Download streamlit using this command**:
      ```
      pip install streamlit
      ```
      
   c. **Download 2 Datasets (IRIS, Digits) from Scikit-learn**:
      ```python
      from sklearn.datasets import load_iris, load_digits
      iris = load_iris()
      digits = load_digits()
      ```
      
   d. **Import 3 required ML Classifiers using from sci-kit learn**:
      ```python
      from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
      from sklearn.svm import SVC
      ```

## 5.2. Loading Datasets

   1. **Load the datasets from the imported sci-kit learn library "datasets"**:
      ```python
      iris_data = datasets.load_iris()
      digits_data = datasets.load_digits()
      ```

## 5.3. Assigning Classifiers

   1. **Create a dictionary for 3 classifiers key as name, value as classifier calling method**:
      ```python
      classifiers = {
          'Random Forest': RandomForestClassifier(),
          'Gradient Boosting': GradientBoostingClassifier(),
          'Support Vector Machine': SVC()
      }
      ```


5.4. Adding a Radio button for choosing Dataset

5.5. Splitting Data for every model in an 80:20 Ratio with random seed as 42

5.6. Choosing Classifier by using the drop down option

5.7. Giving Input Feature values by using Horizontal bars

5.8. Prediction of Classes using Predict functionality + Creating Radio Button 

5.9. Adding HTML & CSS for Prediction Button 


6. Running Web Application 

	1. copy the streamlit.py or your .py extension python file 
	2. use this command to run the application 
              streamlit run YOURFILEPATH
	      streamlit run streamlit.py (my code)


7. Cross Checking Values in Web Applications 

	1. Choose Dataset
	2. Choose Classifier
	3. Change Input Features as per your requirement
	4. Click the Predict button
	5. Observe Predictive class in the Application
