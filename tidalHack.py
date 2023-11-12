import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:\\Users\\midde\\Downloads\\StudentsPerformance_with_headers.csv")

scaler = StandardScaler()
data[['Weekly study hours', 'GRADE']] = scaler.fit_transform(data[['Weekly study hours', 'GRADE']])

x = data[['Weekly study hours']]
y = data['GRADE']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)

alphas = np.logspace(-4, 4, 100)
lasso_cv = LassoCV(alphas = alphas, cv = 5)

lasso_cv.fit(xTrain, yTrain)

optimal = lasso_cv.alpha_

lasso_model =Lasso(alpha = optimal)
lasso_model.fit(xTrain, yTrain)

yPred = lasso_model.predict(xTest)

plt.figure(figsize=(10, 6))
sns.scatterplot(x = yTest, y = yPred)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Lasso Regression: Actual vs. Predicted Values")
plt.show()

#######################################################
#Using Linear Regression Model:

#data = pd.read_csv("C:\\Users\\midde\\Downloads\\StudentsPerformance_with_headers.csv")

#print(data.head())
#print(data.isnull().sum())

#model = LinearRegression()
#model.fit(xTrain, yTrain)
#yPred = model.predict(xTest)

#mse = mean_squared_error(yTest, yPred)
#r2 = r2_score(yTest, yPred)
#print(mse)
#print(r2)
#######################################################

#Predicting new values


#import pandas as pd
#from sklearn.linear_model import LinearRegression

#data = pd.read_csv("your_dataset.csv")

#X = data[['Weekly study hours']]
#y = data['GRADE']

#linear_model = LinearRegression()
#linear_model.fit(X, y)

#new_students = pd.DataFrame({'study_hours': [5, 8, 10]})

#predicted_grades = linear_model.predict(new_students)

#print("Predicted Grades for New Students:")
#for i, grade in enumerate(predicted_grades):
#    print(f"Student {i+1}: {grade}")