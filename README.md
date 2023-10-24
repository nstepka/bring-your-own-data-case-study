Bring Your Own Data Case Study: Interactive Data Analysis and Causation Tool
This project presents an interactive web application built using Streamlit, designed to empower users to upload a dataset, perform advanced data analysis, regression analysis, classification tasks, extensive data exploration, and even causation analysis.

Features
Data Uploading and Preprocessing
Upload CSV File: Users can upload their own CSV dataset.
Data Preview: The uploaded data is displayed for a quick overview.
Handle Missing Values: Missing data can be handled using various methods.
Currency and Percentage Changes: Convert currency and percentage columns to numeric format.
Drop Columns: Easily drop unnecessary columns.
Data Transformation: Apply various data transformations.
Data Encoding: Encode categorical variables using one-hot encoding.
Time Series Features: Convert date/time columns and create new features.
Binning/Bucketing: Discretize numeric variables into bins.
Custom Feature Engineering: Create new features using custom logic.
Aggregate Columns: Perform aggregations and create new columns.
Create Binary Flags: Generate binary flags for specified conditions.
Data Exploration
Boxplot Visualization: Visualize the distribution of numeric data using boxplots.
Binary Distribution: Examine binary distribution using count plots.
Feature Comparison Graphs: Compare two variables using scatter and line plots.
Feature Importance: Evaluate feature importance for regression and classification tasks.
Regression Analysis
Regression Models: Train and compare various regression models:
Gradient Boosting Regressor
Random Forest Regressor
Linear Regression
Decision Tree Regressor
Model Evaluation: Compare models using R-squared scores and Mean Absolute Error (MAE).
Feature Importance: Visualize feature importance.
Heat Map: Display a heatmap to visualize correlations.
Prediction vs. Actual: Compare predictions against actual values.
Residual Plot: Examine residuals to assess model performance.
Classification Analysis
Classification Models: Train and compare various classification models.
Model Evaluation: Evaluate models using accuracy, precision, recall, and F1-score.
Feature Importance: Visualize feature importance.
Heat Map: Display a heatmap to visualize correlations.
Prediction vs. Actual: Compare predictions against actual labels.
Extensive Data Analysis
Dataset: Explore the Iris dataset for clustering, dimensionality reduction, and feature selection.
Clustering: Visualize clusters in 3D space using K-Means clustering.
Dimensionality Reduction: Reduce dimensionality using PCA and visualize in 2D space.
Feature Selection: Select top features based on F-values and p-values.
Causality Analysis
Define Relationships: Define causal relationships between variables using a graphical interface or upload a DOT file.
Create Causal Model: Create a causal model based on defined relationships and estimate causal effects.
Estimation: Estimate causal effects and assess their significance.
Run Refutation Tests: Perform refutation tests to test the reliability of causal estimates.
Usage
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/interactive-data-analysis-tool.git
Change into the project directory:

bash
Copy code
cd interactive-data-analysis-tool
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
The app will open in your default web browser, allowing you to interact with the tool.

Contribution
Contributions to this project are welcome. To contribute, follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them with descriptive commit messages.
Push your changes to your forked repository.
Create a pull request to the original repository, describing your changes and their purpose.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This tool was built using Python, Streamlit, scikit-learn, pandas, plotly, graphviz, and causalgraphicalmodels.
Special thanks to the open-source community for their contributions to the libraries used in this project.
