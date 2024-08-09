# Pejmanai Data Analysis Package

![Package Banner](https://i.postimg.cc/s2K9BMDh/design.png)


## Overview

* `pejmanai_data_analysis` is a Python package for comprehensive data analysis, including read data in csv format, data preprocessing, data visualization, and machine learning modeling for both regression and classification problems. It provides tools to streamline the process of understanding and modeling datasets with ease.

## Features

- **Data Reading:** Load datasets from CSV files with error handling.
- **Data Description:** View basic statistics, data information, and missing values.
- **Data Preprocessing:** Handle missing values and encode categorical variables.
- **Data Visualization:** Generate scatter plots, histograms, KDE plots, and heatmaps.
- **Regression Models:** Evaluate multiple regression models including Linear Regression, Ridge Regression, Decision Trees, Random Forests, and K-Nearest Neighbors.
- **Classification Models:** Compare various classification models such as Decision Trees, Random Forests, Support Vector Machines, K-Nearest Neighbors, and MLP Classifiers.

## Installation

* You can install the package using pip:
`pip install pejmanai_data_analysis`

## Usage
* Data Reading
- `from pejmanai_data_analysis.app import read_csv`
- `df = read_csv('path/to/your/data.csv')`
- `print(df)`

* Data Description
- `from pejmanai_data_analysis.app import data_description`
- `data_description('path/to/your/data.csv')`

* Data Preprocessing
- `from pejmanai_data_analysis.app import data_preprocessing`
- `df_preprocessed = data_preprocessing('path/to/your/data.csv')`
- `print(df_preprocessed)`

* Data Visualization
- `from pejmanai_data_analysis.app import data_visualization`
- `data_visualization('path/to/your/data.csv', 'x_column', 'y_column')`

* Data Prediction (Regression)
- `from pejmanai_data_analysis.app import data_prediction`
- `data_prediction('path/to/your/data.csv', 'target_column')`

* Data Classification
- `from pejmanai_data_analysis.app import data_classification`
- `data_classification('path/to/your/data.csv', 'target_column')`

## License
* This project is licensed under the MIT License

## Contact
For any questions or feedback, please reach out to [Pejman Ebrahimi](https://www.giltech-megoldasok.com/).
