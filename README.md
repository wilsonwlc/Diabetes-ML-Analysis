# Diabetes and Lifestyle Factors - Machine Learning Analysis

## Project Description
* To better understand the relationship between lifestyle and diabetes in the US

## Dataset Information
### Target
* `Diabetes_binary`: 0 = no diabetes; 1 = pre-diabetes or diabetes
### Features
1.	`HighBP`: 0 = no high blood pressure; 1 = high blood pressure
2.	`HighChol`: 0 = no high cholesterol; 1 = high cholesterol
3.	`CholCheck`: 0 = no cholesterol check in 5 years; 1 = cholesterol check in 5 years
4.	`BMI`: Body Mass Index (integer)
5.	`Smoker`: 0 = no; 1 = yes
6.	`Stroke`: 0 = no; 1 = yes
7.	`HeartDiseaseorAttack`: 0 = no; 1 = yes
8.	`PhysActivity`: 0 = no; 1 = yes
9.	`Fruits`: 0 = no; 1 = yes (consume fruit daily)
10.	`Veggies`: 0 = no; 1 = yes (consume vegetables daily)
11.	`HvyAlcoholConsump`: 0 = no; 1 = yes (heavy alcohol consumption)
12.	`AnyHealthcare`: 0 = no; 1 = yes (healthcare coverage)
13.	`NoDocbcCost`: 0 = no; 1 = yes (couldn’t see doctor due to cost)
14.	`GenHlth`: General health (scale 1-5, from excellent to poor)
15.	`MentHlth`: Days in the past month with poor mental health (1-30)
16.	`PhysHlth`: Days in the past month with poor physical health (1-30)
17.	`DiffWalk`: 0 = no; 1 = yes (difficulty walking)
18.	`Sex`: 0 = female; 1 = male
19.	`Age`: Age in categories (1-13, from 18-24 to 80+)
20.	`Education`: Education level (1-6, from no school to college graduate)
21.	`Income`: Income level (1-8, from <$10,000 to $75,000+)

## Project Structure
* `main.ipynb`: Jupyter Notebook that implements the ANN for diabetes risk prediction
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

## Results
The following variables have high feature importance: `GenHlth`, `BMI_scaled`, `Age`, `HighBP` and `HighChol`
