# Diabetes and Lifestyle Factors - Machine Learning Analysis

## Project Description
* This project aims to better understand the relationship between lifestyle and diabetes in the US

## Dataset Information
The data is downloaded from Centers for Disease Control and Prevention (CDC) on [Kaggle](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system?select=2015.csv). The schema can be found [here](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).
### Target
* `Diabetes_binary`: 0 = no diabetes; 1 = pre-diabetes or diabetes
### Features
1.	`HighBP`: 0 = no high blood pressure; 1 = high blood pressure
2.	`HighChol`: 0 = no high cholesterol; 1 = high cholesterol
3.	`CholCheck`: 0 = no cholesterol check in 5 years; 1 = cholesterol check in 5 years
4.	`BMI`: Body Mass Index (integer)
5.	`Smoker`: Have you smoked at least 100 cigarettes in your entire life? 0 = no; 1 = yes
6.	`Stroke`: (Ever told) you had a stroke. 0 = no; 1 = yes
7.	`HeartDiseaseorAttack`: Coronary heart disease (CHD) or myocardial infarction (MI) 0 = no; 1 = yes
8.	`PhysActivity`: Physical activity in past 30 days - not including job 0 = no; 1 = yes
9.	`Fruits`: Consume Fruit 1 or more times per day 0 = no; 1 = yes
10.	`Veggies`: Consume Vegetables 1 or more times per day 0 = no; 1 = yes
11.	`HvyAlcoholConsump`: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no; 1 = yes
12.	`AnyHealthcare`: Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no; 1 = yes
13.	`NoDocbcCost`: Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no; 1 = yes
14.	`GenHlth`: Would you say that in general your health is (scale 1-5): 1 = excellent; 2 = very good; 3 = good; 4 = fair; 5 = poor
15.	`MentHlth`: Regarding your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? Scale 1-30 days
16.	`PhysHlth`: Regarding your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? Scale 1-30 days
17.	`DiffWalk`: Do you have serious difficulty walking or climbing stairs? 0 = no; 1 = yes
18.	`Sex`: 0 = female; 1 = male
19.	`Age`: Age in categories. 1 = 18-24; 2 = 25-29; ...; 13 = 80 or older
20.	`Education`: Education level. 1 = Never attended school or only kindergarten; 2 = Grades 1 through 8 (Elementary); 3 = Grades 9 through 11 (Some high school); 4 = Grade 12 or GED (High school graduate); 5 = College 1 year to 3 years (Some college or technical school); 6 = College 4 years or more (College graduate)
21.	`Income`: Income level. 1 = less than $10,000; 2 = $10,000 to less than $15,000;...; 8 = $75,000 or more

## Project Structure
* `data/` 
    * `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`: Cleaned dataset downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) for first round of analysis
    * `BRFSS-2015_binary.csv`: Raw data for the current round of analysis
    * `BRFSS-2015_binary_clean.csv`: Preprocessed dataset for current round of analysis
    * `data_train_borderline_smote.csv`, `data_val.csv`, `data_test.csv`: Training set, validation set and test set for modelling in current round of analysis
* `schema/`:
    `codebook15_llcp.pdf`: Schema for 2015 data
* `method/`: Technical documentation folder containing explanations of methodologies used in this project including borderline SMOTE, Neural network, XGBoost, Permutation feature importance
* `archive/Result - Neural network.ipynb`: 1st round of analysis implementing the ANN for diabetes risk prediction. It uses the dataset downloaded from `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
* `2.0 Data source.ipynb`: Notebook detailing data acquisition process and initial dataset information
* `2.1 Data checking.ipynb`:  Initial data quality checks, missing value analysis, and basic data validation
* `2.2 Data exploration.ipynb`: Exploratory data analysis with visualizations of feature distributions and relationships
* `2.3 Data preprocessing.ipynb`: Data cleaning, transformation, and feature engineering steps
* `3.1 Model building - NN.ipynb`: Implementation of neural network models with hyperparameter tuning
* `3.2 Model building - xgboost.ipynb`: Implementation of XGBoost models with hyperparameter optimization


