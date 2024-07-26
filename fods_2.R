data1<-read.csv(file.choose(ADITYA.CSV),header = T)
#import a dataset
#'Aditya' is a dataset used for this example
#display the first few rows of the dataset
head(data1)
#identify the features of the dataset 
features<-names(data1)
print ("features in the dataset:")
print(features)
#identify the datatype of each feature
datatypes<-sapply(data1,class)
print("Data types of each featurs:")
print(datatypes)
#it gives the number summary of the features
summary_Aditya<-summary(data1)
print("Number summar of the features")
print(summary.Aditya)
#brief explanation and potential actions 
explanation<-"The Aditya dataset has 150 observation with 5 features :sepal.length,sepal.width,petal.length,petal.width, and species the first four feautures are numerically representing measurement in centemeter while the species features is a factor with three level sentosa,verselicolor and virginical from the summary can be observed . the range , median , and quartiles for the numeric features , indicating the central tendency of the measurement "
cat(explanation)
data<-"As a data scientist potential actions can include visualizing the data to understand relationships and distribution performing the exploratory data analysis to detect pattern anomalies .Applaying machine learning model to classify the species based on the measurements .Using statistical tests to compare species based on measurement .Using statistical tasks to compare species measurement and derive meaningfull conclusions"
cat(data)