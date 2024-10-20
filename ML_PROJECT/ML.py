"""
                                                                            work flow

                            collect_data -> data_processing -> train_test_split -> logistic_Regression_model

        Note:
            We are using this because logistic classification model work well with binary data

                                                                            Training
                                                                            
                                                                    (supervised learning)
                                                                    
                            fetch_data -> training_logistic_regression_model -> predict
"""

import numpy as np #Used for making numpy array
import pandas as pd #Used for loading dataframe
from sklearn.model_selection import train_test_split  #This is used to seperate our data set into test and train dataset
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score
sonar_data=pd.read_csv("C:\\Users\LENOVO\Desktop\ML_PROJECT\sonar data.csv",header=None)
sonar_data.head()
sonar_data.describe()
sonar_data[60].value_counts()
#to get mean value for all column
sonar_data.groupby(60).mean()
#try to seperate data and lables
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
#split data into training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
#Model training with logistic regression
model=LogisticRegression()
#Training  the Logistic regression model with training data
model.fit(X_train,Y_train)
#Finding the accuracy of the model
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy : ",training_data_accuracy*100)
#find the accuracy of the test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy of test data: ",test_data_accuracy*100)
#Making a predictive system
input_data=(0.02,0.0371,0.0428,0.0207,0.0954,0.0986 ,0.1539 ,0.1601 ,0.3109	,0.2111,0.1609,0.1582,0.2238,0.0645 ,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032

)
input_data_as_numpy_array=np.asarray(input_data)
#rename as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]=='R'):
    print("The object is rock")
else:
    print("The object is mine")
