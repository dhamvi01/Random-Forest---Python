import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Taregt'] = iris.target

print(data)
#print(data.head())

x = data.iloc[:,0:4]
y = data[['Taregt']]

model = RandomForestClassifier()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)
#print(x.head())
#print(y.head())

model.fit(x_train,y_train)
pred = model.predict(x_test)
print(model.score(x,y))
print(confusion_matrix(y_test,pred))
