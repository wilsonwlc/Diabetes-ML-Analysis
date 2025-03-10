# Diabetes and Lifestyle Factors - Machine Learning Analysis

## Project Description
* To better understand the relationship between lifestyle and diabetes in the US

## Dataset Information
The dataset is downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). The schema can be download from [Centers for Disease Control and Prevention (CDC)](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf). Note that the data source does not specify the type of diabetes, whether it's type 1 or type 2.
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
15.	`MentHlth`: Regarding your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days
16.	`PhysHlth`: Regarding your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days
17.	`DiffWalk`: Do you have serious difficulty walking or climbing stairs? 0 = no; 1 = yes
18.	`Sex`: 0 = female; 1 = male
19.	`Age`: Age in categories. 1 = 18-24; 2 = 25-29; ...; 13 = 80 or older
20.	`Education`: Education level. 1 = Never attended school or only kindergarten; 2 = Grades 1 through 8 (Elementary); 3 = Grades 9 through 11 (Some high school); 4 = Grade 12 or GED (High school graduate); 5 = College 1 year to 3 years (Some college or technical school); 6 = College 4 years or more (College graduate)
21.	`Income`: Income level. 1 = less than $10,000; 2 = $10,000 to less than $15,000;...; 8 = $75,000 or more

## Project Structure
* `Result - Neural network.ipynb`: Jupyter Notebook that implements the ANN for diabetes risk prediction
*  `data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv`: Dataset containing lifestyle and medical history data

## Data Preprocessing
### Feature Engineering
* Discretises the numerical `PhysHlth` and `MentHlth` into 0, 1-7, 8-14, 15-21, 22-28, 29-30

## Modeling
* A simple neural network model was trained. After training, the training accuracy and validation accuracy were used to evaluate the model’s performance. 
* To reduce avoidable bias i.e. the gap between Bayes error and training error, a larger network was fitted. Random Search was used to perform the hyperparameter search with the objective of maximising accuracy. 
* To reduce variance, L2 regularisation was added. Another Random Search was run for 10 trials, now optimising the validation accuracy with L2 regularisation hyperparameters. 
* After running the tuning process, the best model was evaluated for test accuracy. 
* To interpret the model, permutation feature importance was computed and visualised.


