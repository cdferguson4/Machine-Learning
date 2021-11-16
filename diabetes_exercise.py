''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''
from sklearn.datasets import load_diabetes
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

diabetes = load_diabetes()
#how many sameples and How many features?
#print(diabetes.data.shape)
############## 442 samples and 10 features#######################

# What does feature s6 represent?
#print(diabetes.DESCR)
### It represents Blood Sugar level#######################################

#print(diabetes)


#print out the coefficient
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    diabetes.data,diabetes.target, random_state=11
)

from sklearn.linear_model import LinearRegression
#THree steps to model something with sklearn
#1. Set up model
lr = LinearRegression()
#2. Use fit to train our model
lr.fit(X=x_train,y=y_train)
#print the coefficient
coef = lr.coef_


print(coef)

#print out the intercept
intercept = lr.intercept_
print(intercept)

#3.Use predict to test your model
predicted = lr.predict(x_test)

expected = y_test

print(predicted[:15])
print(expected[:15])


# create a scatterplot with regression line
plt.plot(expected, predicted, ".")


x = np.linspace(0,330,100)
print(x)
y=x
plt.plot(x,y)

plt.show()