import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(df.head())# Display the First Few Rows
#Summary Statistics
print(df.describe())
#Information About Data Types and Missing Values
print(df.info())
#Check for Missing Values
print(df.isnull().sum())
#Class Distribution
print(df['species'].value_counts())
#Correlation Matrix
correlation_matrix = df.iloc[:, :-1].corr()
print(correlation_matrix)
#Pairplot of the Features
sns.pairplot(df, hue='species')
plt.show()
#Histogram for Each Feature
df.hist(figsize=(10, 8), bins=30)
plt.show()
#Boxplot of Features by Species
for feature in iris.feature_names:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Species')
    plt.show()
#Violin Plot of Features by Species
for feature in iris.feature_names:
    plt.figure(figsize=(10, 4))
    sns.violinplot(x='species', y=feature, data=df)
    plt.title(f'Violin Plot of {feature} by Species')
    plt.show()
#Scatter Plot Matrix
sns.pairplot(df, hue='species')
plt.show()
#Feature Importance using Random Forest
X = df.iloc[:, :-1]
y = df['species']
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)
importances = clf.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=iris.feature_names, y=importances)
plt.title('Feature Importance')
plt.show()
#Principal Component Analysis (PCA) and Plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['species'] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df)
plt.title('PCA of Iris Dataset')
plt.show()
#Distribution of Each Feature
for feature in iris.feature_names:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.show()
#Heatmap of Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
scaled_df['species'] = y

print(scaled_df.head())
#Boxplot of Scaled Features
for feature in iris.feature_names:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='species', y=feature, data=scaled_df)
    plt.title(f'Boxplot of Scaled {feature} by Species')
    plt.show()
#Calculate Z-scores
z_scores = np.abs(stats.zscore(X))
print(pd.DataFrame(z_scores, columns=iris.feature_names).describe())
#Pairwise Distance Matrix
dist_matrix = pairwise_distances(X)
print(pd.DataFrame(dist_matrix).describe())
#Feature Pairwise Scatter Plots
for i in range(len(iris.feature_names)):
    for j in range(i + 1, len(iris.feature_names)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df.iloc[:, i], y=df.iloc[:, j], hue=df['species'])
        plt.xlabel(iris.feature_names[i])
        plt.ylabel(iris.feature_names[j])
        plt.title(f'Scatter Plot of {iris.feature_names[i]} vs {iris.feature_names[j]}')
        plt.show()
#KDE Plot for Each Feature
for feature in iris.feature_names:
    plt.figure(figsize=(10, 4))
    sns.kdeplot(df[feature], hue=df['species'], fill=True)
    plt.title(f'KDE Plot of {feature}')
    plt.show()
#Pairwise KDE Plots
sns.pairplot(df, kind='kde', hue='species')
plt.show()
#Bar Plot of Mean Feature Values by Species
mean_values = df.groupby('species').mean()
mean_values.plot(kind='bar', figsize=(10, 6))
plt.title('Mean Feature Values by Species')
plt.show()
#Feature Variance
feature_variance = df.iloc[:, :-1].var()
print(feature_variance)
#Class-wise Feature Boxplots
for species in df['species'].unique():
    subset = df[df['species'] == species]
    for feature in iris.feature_names:
        plt.figure(figsize=(10, 4))
        sns.boxplot(y=feature, data=subset)
        plt.title(f'Boxplot of {feature} for {species}')
        plt.show()

