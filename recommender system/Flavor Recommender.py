import pandas as pd
import numpy as np

#file of data of age,gender,Flavor
df = pd.read_csv("flv.csv")
print(df)

print(df.shape) # the number of the raws(20) and cols(3)

print(df.head(10)) #print the first 10
print(df.tail(10)) #the last 10

print(df.count()) #how much from each kind
print(df.dtypes) #the type od the objects



from sklearn.preprocessing import LabelEncoder

Encd = LabelEncoder() #serlizable to the data for the training set
#gender
print(Encd.fit(['Male','Female']))

df['Gender'] = Encd.transform(df['Gender'])

x = df.drop(columns= ['Flavour']) #the table without the flav
y = df.drop(columns=['Age','Gender']) #the table without the age and gender

print(x)
print(y)

from sklearn.tree import DecisionTreeClassifier

Model = DecisionTreeClassifier()
Model.fit(x,y)
print(Model)

#test
age = 17
gender = Encd.transform(['Male'])

print(Model.predict([[age,gender]]))


def predict():
    age = input('Age: ')
    genderIn = input('Gender: ')
    gender = Encd.transform([genderIn])
    prediction = Model.predict([[age,gender]])
    print("Recommended Flavor: ", prediction)

if __name__ == '__main__':
    predict()
