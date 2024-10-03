import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data train.csv
df = pd.read_csv(
    "train.csv"
    )

# Display the first 5 rows of the data
df.head()
df.info()

#Remove the columns that are not needed
df.drop(['Cabin', 'Name'], axis=1, inplace=True)

# sns.pairplot(df[['Transported','HomePlanet']],dropna=True)

# sns.pairplot(df[['Transported','CryoSleep']],dropna=True)

# sns.pairplot(df[['Transported','Cabin']],dropna=True)

# sns.pairplot(df[['Transported','Destination']],dropna=True)

# sns.pairplot(df[['Transported','Age']],dropna=True) 

# sns.pairplot(df[['Transported','VIP']],dropna=True)

# sns.pairplot(df[['Transported','RoomService']],dropna=True)

# sns.pairplot(df[['Transported','FoodCourt']],dropna=True)

# sns.pairplot(df[['Transported','ShoppingMall']],dropna=True)

# sns.pairplot(df[['Transported','Spa']],dropna=True)

# sns.pairplot(df[['Transported','VRDeck']],dropna=True)

df.columns

# sns.pairplot(df[['Survived','Age']],dropna=True)

# sns.pairplot(df[['Survived','Parch']],dropna=True)

# 分群
df.groupby('Transported').mean(numeric_only=True)

df['HomePlanet'].value_counts()
df['Destination'].value_counts()
df['VIP'].value_counts()

df.isnull().sum().sort_values(ascending=False)


#FIll the missing values with the median of the column
#1.Age
df.groupby('Transported')['Age'].median()
df.groupby('Transported')['Age'].transform("median")
df['Age'].isnull().value_counts()

df['Age']=df['Age'].fillna(df.groupby('Transported')['Age'].transform('median'))    
df['Age'].isnull().value_counts()

df.isnull().sum().sort_values(ascending=False)
#2.RoomService
df.groupby('Transported')['RoomService'].median()
df.groupby('Transported')['RoomService'].transform("median")
df['RoomService'].isnull().value_counts()

df['RoomService']=df['RoomService'].fillna(df.groupby('Transported')['RoomService'].transform('median'))    
df['RoomService'].isnull().value_counts()
#3.FoodCourt
df.groupby('Transported')['FoodCourt'].median()
df.groupby('Transported')['FoodCourt'].transform("median")
df['FoodCourt'].isnull().value_counts()

df['FoodCourt']=df['FoodCourt'].fillna(df.groupby('Transported')['FoodCourt'].transform('median'))
df['FoodCourt'].isnull().value_counts()

#4.Spa
df.groupby('Transported')['Spa'].median()
df.groupby('Transported')['Spa'].transform("median")
df['Spa'].isnull().value_counts()

df['Spa']=df['Spa'].fillna(df.groupby('Transported')['Spa'].transform('median'))
df['Spa'].isnull().value_counts()

#5.VRDeck
df.groupby('Transported')['VRDeck'].median()
df.groupby('Transported')['VRDeck'].transform("median")
df['VRDeck'].isnull().value_counts()

df['VRDeck']=df['VRDeck'].fillna(df.groupby('Transported')['VRDeck'].transform('median'))
df['VRDeck'].isnull().value_counts()

#6.ShoppingMall
df.groupby('Transported')['ShoppingMall'].median()
df.groupby('Transported')['ShoppingMall'].transform("median")
df['ShoppingMall'].isnull().value_counts()

df['ShoppingMall']=df['ShoppingMall'].fillna(df.groupby('Transported')['ShoppingMall'].transform('median'))
df['ShoppingMall'].isnull().value_counts()

df.isnull().sum().sort_values(ascending=False)

#Fill the missing values with the mode of the column
#7.CryoSleep
df['CryoSleep'].value_counts().idxmax()
df['CryoSleep'].fillna(df['CryoSleep'].value_counts().idxmax(), inplace=True)

#8 HomePlanet
df['HomePlanet'].value_counts().idxmax()
df['HomePlanet'].fillna(df['HomePlanet'].value_counts().idxmax(), inplace=True) 

#9 Destination
df['Destination'].value_counts().idxmax()
df['Destination'].fillna(df['Destination'].value_counts().idxmax(), inplace=True)   

#10 VIP
df['VIP'].value_counts().idxmax()
df['VIP'].fillna(df['VIP'].value_counts().idxmax(), inplace=True)   

df.isnull().sum().sort_values(ascending=False)  
df.info()
df.isnull().sum() 

#Create dummy column 
df = pd.get_dummies(data=df, 
                    dtype=int,
                    columns=['CryoSleep','HomePlanet','Destination','VIP'])

df.head()

#Drop redudant column
df.drop('CryoSleep_False', axis=1, inplace=True)
df.drop('VIP_False', axis=1, inplace=True)
df.drop('HomePlanet_Earth', axis=1, inplace=True)
df.drop('Destination_55 Cancri e', axis=1, inplace=True)

df.info()

df.corr()
#Create a CSV file corr.csv and export the df.corr() to corr.csv
df.corr().to_csv('corr.csv')



X = df.drop(['Transported'], axis=1)
y = df['Transported']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=67)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1900)
lr.fit(X_train, y_train)    

preditions = lr.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

accuracy_score(y_test, preditions)
recall_score(y_test, preditions)    
precision_score(y_test, preditions) 

confusion_matrix(y_test, preditions)

pd.DataFrame(confusion_matrix(y_test, preditions), 
             columns=['Predicted not Transported', 'Predicted Transported'],
             index=['True not Transported', 'True Transported'])


#Model Export   
import joblib
joblib.dump(lr, 'spaceship-titanic-LR-20241002.pkl', compress=3)      

print(accuracy_score(y_test, preditions))
print(recall_score(y_test, preditions)) 
print(precision_score(y_test, preditions))  

# F1 Score
print(f1_score(y_test, preditions))