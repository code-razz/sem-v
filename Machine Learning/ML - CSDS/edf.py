import pandas as pd
data=pd.read_csv("C:\\Users\\aadar\\Downloads\\ds4.csv")
prob_age = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2}
#Gender: Male:0, Female:1
prob_gender = {0: 0.4, 1: 0.6}
#Family History: Yes:1, No:0
prob_family = {0: 0.7, 1: 0.3}
#Diet: High:0, Medium:1
prob_diet = {0: 0.5, 1: 0.5}
#Lifestyle: Athlete:0, Active: 1, Moderate: 2, Sedentary: 3
prob_lifestyle ={0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}
#Cholesterol: High:0, BorderLine:1, Normal:2
prob_cholesterol = {0: 0.4, 1: 0.3, 2: 0.3}
prob_heart_disease = {
(0, 0, 0, 0, 0, 0): 0.1,
(1, 1, 1, 1, 1, 1): 0.9 # Example conditional probability entries
}
def calculate_probability(age, gender, family, diet, lifestyle, cholesterol):
    key =(age, gender, family, diet, lifestyle, cholesterol)
    if key in prob_heart_disease:
        return prob_heart_disease [key]
    else:
        return 0.5 # Default probability if data is not available
print('For Age enter SuperSeniorCitizen: 0, SeniorCitizen:1, MiddleAged: 2, Youth:3, Teen:4')
print('For Gender enter Male:0, Female:1')
print('For Family History enter Yes:1, No:0')
print('For Diet enter High:0, Medium:1')
print('For LifeStyle enter Athlete:0, Active: 1, Moderate: 2, Sedentary:3')
print('For Cholesterol enter High:0, BorderLine:1, Normal:2')
age = int(input('Enter Age: '))
gender= int(input('Enter Gender: '))
family =int(input('Enter Family History: '))
diet =int(input('Enter Diet: '))
lifestyle= int(input('Enter Lifestyle: '))
cholesterol = int(input('Enter Cholesterol: '))
#Calculate the probability of heart disease given the evidence 
probability =calculate_probability(age, gender, family, diet, lifestyle, cholesterol)
print(f'Probability of Heart Disease: {probability:.2f}')