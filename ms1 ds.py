import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


file_path = r"C:\GIU\Semster 4\Data Science\Project 1 Data Science\ObesityDataSet_raw_and_data_sinthetic.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError("Dataset file not found. Please upload csv to Colab.")

# (a) Display the first and last 12 rows of the dataset
print("First 12 rows:")
print(df.head(12), "\n")
print("Last 12 rows:")
print(df.tail(12), "\n")

# (b) Identify and print the total number of rows and columns present
rows, columns = df.shape
print(f"Total rows: {rows}, Total columns: {columns}\n")

# (c) List all column names along with their corresponding data types
print("Column Names and Data Types:")
print(df.dtypes, "\n")

# (d) Print the name of the first column
print(f"The first column name is: {df.columns[0]}\n")

# (e) Generate a summary of the dataset
print("Dataset Summary:")
print(df.info(), "\n")

# (f) Choose a categorical attribute and display distinct values
category_column = "MTRANS"
print(f"Distinct values in '{category_column}':\n{df[category_column].unique()}\n")

# (g) Identify the most frequently occurring value in the chosen categorical attribute
most_frequent_value = df[category_column].mode()[0]
print(f"The most frequent value in '{category_column}' is: {most_frequent_value}\n")

# (d) Convert a numerical column from integer to string
df["Age"] = df["Age"].astype(str)
print("Converted 'Age' column from integer to string.\n")

# (e) Group dataset based on two categorical features and analyze results
grouped_df = df.groupby(["Gender", "MTRANS"]).size().reset_index(name="Count")
print("Grouped dataset based on 'Gender' and 'MTRANS':")
print(grouped_df.head(), "\n")

# (f) Check for missing values
missing_values = df.isnull().sum()
print("Missing Values in Dataset:")
print(missing_values, "\n")

# (g) Replace missing values with median or mode
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
print("Missing values replaced successfully.\n")

# (h) Divide numerical column into 5 bins
bin_counts = pd.cut(df["Weight"], bins=5).value_counts()
print("Binning 'Weight' column into 5 bins:")
print(bin_counts, "\n")

# (i) Identify row corresponding to maximum value of a numerical feature
max_value_row = df[df["Weight"] == df["Weight"].max()]
print("Row with maximum 'Weight':")
print(max_value_row, "\n")

# (j) Construct boxplot for an attribute
plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Weight"])
plt.title("Boxplot of Weight")
plt.show()

# (k) Generate a histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["Weight"], bins=10, kde=True)
plt.title("Histogram of Weight")
plt.show()

# (l) Create scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Weight"], y=df["Height"])
plt.title("Scatterplot of Weight vs Height")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()

# (m) Normalize numerical attributes
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if "Age" in numerical_columns:
    numerical_columns.remove("Age")  # Exclude 'Age' since it was converted to string
if numerical_columns:  # Ensure there are numerical columns to scale
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print("Numerical attributes normalized.\n")

# (n) Perform PCA
if len(numerical_columns) >= 2:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numerical_columns])
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"])
    plt.title("PCA Result")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# (o) Analyze correlation using a heatmap
if len(numerical_columns) > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# (a) Calculate and display correlation matrix
correlation_matrix = df[numerical_columns].corr()
print("Correlation Matrix:")
print(correlation_matrix, "\n")

# (b) Find class distribution of a categorical feature
class_distribution = df[category_column].value_counts()
print(f"Class Distribution of '{category_column}':")
print(class_distribution, "\n")

# (c) Create new features (Feature Engineering)
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
print("New feature 'BMI' added.\n")
print(df[["Weight", "Height", "BMI"]].head())