# Model Using
import joblib
model_pretrained = joblib.load("spaceship-titanic-LR-20241002.pkl")
import pandas as pd
df_test = pd.read_csv("test.csv")
df_test.head()

#Drop PassengerId and Name and Cabin
df_test.drop("Name", axis=1, inplace=True)
df_test.drop("Cabin", axis=1, inplace=True)
df_test.info()
df_test.isnull().sum()


#Fill the missing values with the mode of the column
df_test['CryoSleep'].fillna(df_test['CryoSleep'].value_counts().idxmax(), inplace=True)
df_test['HomePlanet'].fillna(df_test['HomePlanet'].value_counts().idxmax(), inplace=True)
df_test['Destination'].fillna(df_test['Destination'].value_counts().idxmax(), inplace=True)
df_test['VIP'].fillna(df_test['VIP'].value_counts().idxmax(), inplace=True)

#Fill the missing values with the median of the column
df_test['Age']=df_test['Age'].fillna(df_test.groupby('HomePlanet')['Age'].transform('median'))
df_test['RoomService']=df_test['RoomService'].fillna(df_test.groupby('HomePlanet')['RoomService'].transform('median'))
df_test['FoodCourt']=df_test['FoodCourt'].fillna(df_test.groupby('HomePlanet')['FoodCourt'].transform('median'))
df_test['ShoppingMall']=df_test['ShoppingMall'].fillna(df_test.groupby('HomePlanet')['ShoppingMall'].transform('median'))
df_test['Spa']=df_test['Spa'].fillna(df_test.groupby('HomePlanet')['Spa'].transform('median'))
df_test['VRDeck']=df_test['VRDeck'].fillna(df_test.groupby('HomePlanet')['VRDeck'].transform('median'))

df_test.isnull().sum() 


df_test.head()
#Convert the categorical columns to numerical
df_test = pd.get_dummies(data=df_test, 
                         dtype=int, 
                         columns=["CryoSleep", "HomePlanet", "Destination", "VIP"]) 
df_test.head()

df_test.drop('CryoSleep_False', axis=1, inplace=True)
df_test.drop('VIP_False', axis=1, inplace=True)
df_test.drop('HomePlanet_Earth', axis=1, inplace=True)
df_test.drop('Destination_55 Cancri e', axis=1, inplace=True)


df_test.head()

predictions3 = model_pretrained.predict(df_test)

#Prepare submission file
forSubmissionDF = pd.DataFrame(
    {
        "PassengerId": df_test["PassengerId"],
        "Transported": predictions3
    }
)   

forSubmissionDF.to_csv("for_submission_20241002.csv", index=False)
