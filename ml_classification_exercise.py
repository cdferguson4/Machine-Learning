# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()



#EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up
#print(iris.DESCR)

# display the shape of the data, target and target_names
#print(iris.data.shape)
#print(iris.target.shape)
#figure, axes = plt.subplots(nrows=3,ncols=3,figsize=(3,3))

#for item in zip(axes.ravel(), iris.images ,iris.target):
#    axes,image,target = item
#    axes.imshow(image,cmap=plt.cm.gray_r)
#    axes.set_xticks([])
#    axes.set_yticks([])
#    axes.set_title(target)

#plt.tight_layout()

#plt.show()
# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

# display the values that the model got wrong

# visualize the data using the confusion matrix


##########--------------WHAT PROF DID-------------------------------###################

#print(iris.data.shape)
#print(iris.target.shape)
#print(iris.target_names)

data_train, data_test, target_train,target_test = train_test_split(
    iris.data,iris.target, random_state=11)

print(data_train.shape)
print(target_train.shape)
print(data_test.shape)
print(target_test.shape)

knn= KNeighborsClassifier()

knn.fit(X=data_train,y=target_train)

predicted= knn.predict(X=data_test)

expected = target_test

print(predicted[:20])
print(expected[:20])
print(iris.target_names)

predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[x] for x in expected]

print(predicted[:20])
print(expected[:20])

wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]

print(wrong)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected,y_pred=predicted)
print(confusion)

import pandas as pd
import seaborn as sns

confusion_df = pd.DataFrame(confusion,index=iris.target_names,columns=iris.target_names)

figure = plt.figure()
plt.xlabel("Expected")
plt.ylabel("predicted")
axes = sns.heatmap(confusion_df,
annot=True,cmap=plt.cm.nipy_spectral_r)


plt.show()