STREAMLIT WEB APPLICATION FOR MACHINE LEARNING CLASSIFIERS 

1. Entire Web application is having python,CSS (for styling)

2. In this application we are going to Build 3 Machine Learning Classifiers and Integrating with Webapplication using streamlit (Python framework)

3. ML Classifiers are building on 2 Datasets (IRIS,Digits) downloaded from Sckit learn library


4. Below are the Techstack (frameworks,libraries) we are using in this project
   For Building ML Classfiers,Datasets - Scikit learn 
                   Webapplication      - Streamlit
                   Styling             - CSS


5. Code Walk though / Process : 
	5.1. Import & Download required libraries 
		a. Import & Download required libraries 
		b. Download streamlit using this command "pip install streamlit"
		c. Download 2 Datasets (IRIS,Digits ) from Scikit-learn
		d. Import 3 required ML Classfiers using from Sciki-tlearn
	


5.2. Loading Datasets
	1. Load the datasets from imported scikit learn library "datasets"

5.3. Assigning Classfiers
	1. Create dictionary for 3 classfiers key as name, value as classifier calling method

5.4. Adding Radio button for choosing Dataset

5.5. Splitting Data for every model in 80:20 Ratio with randomseed as 42

5.6. Choosing Classifier by using drop down option

5.7. Giving Input Feature values by using Horizontal bars

5.8. Prdiction of Classes using Predict funtionality + Creating Radio Button 

5.9. Adding HTML & CSS for Prediction Button 


6. Running Web Application 

	1. copy the streamlit.py or your .py extension python file 
	2. use this command to run application 
              streamlit run YOURFILEPATH
	      streamlit run streamlit.py (my code)


7. Cross Checking Values in Web Application 

	1. Choose Dataset
	2. Choose Classfier
	3. Change Input Features as per your requirement
	4. Click Predict button
	5. Observe Predictive class in Application