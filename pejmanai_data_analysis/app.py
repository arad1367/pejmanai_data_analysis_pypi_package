# pejmanai_data_analysis/app.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate
from colorama import Fore, Style
import os

def read_csv(file_path):
    if not os.path.isfile(file_path):
        print(Fore.RED + f"Error: File '{file_path}' does not exist.")
        return None
    try:
        df = pd.read_csv(file_path, encoding='unicode_escape')
        print(Fore.GREEN + f"File '{file_path}' successfully read.")
        return df
    except Exception as e:
        print(Fore.RED + f"Error reading CSV file: {e}")
        return None
    finally:
        print(Style.RESET_ALL)

def data_description(file_path):
    df = read_csv(file_path)
    if df is not None:
        try:
            print(Fore.CYAN + "-"*50 + "\nFirst 10 rows of the data:")
            print(tabulate(df.head(10), headers='keys', tablefmt='fancy_grid'))
            print(Fore.CYAN + "-"*50 + "\nData Description:")
            print(tabulate(df.describe(), headers='keys', tablefmt='fancy_grid'))
            print(Fore.CYAN + "-"*50 + "\nData Information:")
            print(df.info())
            print(Fore.CYAN + "-"*50 + "\nMissing Values:")
            print(df.isna().sum())
        except Exception as e:
            print(Fore.RED + f"Error in data_description: {e}")
        finally:
            print(Style.RESET_ALL)

def data_preprocessing(file_path):
    df = read_csv(file_path)
    if df is not None:
        try:
            df = df.dropna()  # Dropping missing values
            label_encoders = {}
            for column in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = dict(zip(le.classes_, le.transform(le.classes_)))
                print(Fore.GREEN + f"Encoded column '{column}' with labels:")
                print(tabulate(label_encoders[column].items(), headers=['Label', 'Encoded'], tablefmt='fancy_grid'))
            print(Fore.GREEN + "Data preprocessing completed successfully.")
            return df
        except Exception as e:
            print(Fore.RED + f"Error in data_preprocessing: {e}")
        finally:
            print(Style.RESET_ALL)
    return None

def data_visualization(file_path, x_col=None, y_col=None):
    if x_col is None or y_col is None:
        print(Fore.RED + "Error: Missing required parameters x_col and y_col.")
        return
    
    df = read_csv(file_path)
    if df is not None:
        try:
            # Check if specified columns are numeric
            if not np.issubdtype(df[x_col].dtype, np.number) or not np.issubdtype(df[y_col].dtype, np.number):
                print(Fore.RED + "Please ensure both x_col and y_col are numerical columns.")
                return

            plt.figure(figsize=(16, 12))

            # Scatter plot with random colors
            plt.subplot(2, 2, 1)
            colors = np.random.rand(len(df))
            plt.scatter(df[x_col], df[y_col], c=colors, alpha=0.7, cmap='viridis')
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True)

            # Histogram with KDE
            plt.subplot(2, 2, 2)
            df[y_col].plot(kind='hist', bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.7)
            df[y_col].plot(kind='kde', color='red')
            plt.title(f'Histogram and KDE of {y_col}')
            plt.xlabel(y_col)
            plt.ylabel('Density')
            plt.grid(True)

            # Heatmap of correlations
            plt.subplot(2, 2, 3)
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Heatmap of Correlations')

            # Correlation plot of selected features
            plt.subplot(2, 2, 4)
            corr_features = df[[x_col, y_col]].corr()
            sns.heatmap(corr_features, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap of Selected Features')

            plt.tight_layout(pad=3.0)
            plt.show()
        except Exception as e:
            print(Fore.RED + f"Error in data_visualization: {e}")
        finally:
            print(Style.RESET_ALL)

def data_prediction(file_path, target_column):
    df = data_preprocessing(file_path)
    if df is not None:
        try:
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor()
            }

            results = []

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append({
                    'Model': name,
                    'R^2 Score': r2_score(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
                })

            best_model = max(results, key=lambda x: x['R^2 Score'])

            print(Fore.CYAN + "-"*50 + "\nModel Comparison for Regression:")
            print(tabulate(results, headers='keys', tablefmt='fancy_grid'))
            print(Fore.GREEN + "-"*50 + f"\nBest Model: {best_model['Model']} with R^2 Score: {best_model['R^2 Score']:.2f}")

        except Exception as e:
            print(Fore.RED + f"Error in data_prediction: {e}")
        finally:
            print(Style.RESET_ALL)


def data_classification(file_path, target_column):
    df = data_preprocessing(file_path)
    if df is not None:
        try:
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Support Vector Machines': SVC(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42)
            }

            results = []

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                })

            best_model = max(results, key=lambda x: x['Accuracy'])

            print(Fore.CYAN + "-"*50 + "\nModel Comparison for Classification:")
            print(tabulate(results, headers='keys', tablefmt='fancy_grid'))
            print(Fore.GREEN + "-"*50 + f"\nBest Model: {best_model['Model']}")

        except Exception as e:
            print(Fore.RED + f"Error in data_classification: {e}")
        finally:
            print(Style.RESET_ALL)
