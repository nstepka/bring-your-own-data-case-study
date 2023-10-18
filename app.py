import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import matplotlib.pyplot as plt

def evaluate_model_page():
    st.write("Data Preview:")
    st.write(st.session_state.data.head(10))
    st.write("Evaluate the Model")
    
    target_col = st.selectbox("Select the target column", st.session_state.data.columns)

    if st.button("Train and Evaluate Models"):
        X_train, X_test, y_train, y_test = prepare_data(st.session_state.data, target_col)
        models = create_models()
        models = fit_models(X_train, y_train, models)
        results_df = evaluate_models(X_test, y_test, models)
           
        # Display model results in Streamlit
        st.write("Model Evaluation Results:")
        st.write(results_df)

        # Plot model performance
        st.write("Model Performance Comparison:")
        st.pyplot(plot_model_performance(results_df))


def prepare_data(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_models():
    return {
        'GradientBoosting': GradientBoostingRegressor(),
        'RandomForest': RandomForestRegressor(),
        'Linear': LinearRegression(),
        'HistGradientBoosting': HistGradientBoostingRegressor(),
        'DecisionTree': DecisionTreeRegressor(),
        'XGBoost': xgb.XGBRegressor(),
    }

def fit_models(X_train, y_train, models):
    for name, model in models.items():
        model.fit(X_train.astype(float), y_train.astype(float))
    return models

def evaluate_models(X_test, y_test, models):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test.astype(float))

        r2 = r2_score(y_test.astype(float), y_pred)
        mse = mean_squared_error(y_test.astype(float), y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test.astype(float), y_pred)

        results.append({
            'Model': name,
            'R2': r2,
            'RMSE': rmse,
            'MSE': mse,
            'MAE': mae
        })
    return pd.DataFrame(results)

def plot_model_performance(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Model Performance Comparison')

    metrics = ['R2', 'RMSE', 'MSE', 'MAE']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=None)
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig



def sanitize_string(s):
    """Removes non-UTF-8 characters from a string."""
    return s.encode('utf-8', 'ignore').decode('utf-8')


def display_handle_missing_values():
    # Step 3: Check for missing values
    missing_data = st.session_state.data.isnull().sum()
    missing_columns = missing_data[missing_data > 0]

    if not missing_columns.empty:
        missing_values_placeholder = st.empty()
        missing_values_placeholder.write("Columns with missing values:")
        missing_values_placeholder.write(missing_columns)

        # Step 4: Handle missing values
        column_to_handle = st.selectbox("Select a column to handle:", missing_columns.index)
        action = st.selectbox("Choose an action:", ["Fill missing values", "Drop column", "Drop rows with missing values"])

        fill_method = None
        if action == "Fill missing values":
            fill_method = st.selectbox("Select a method to fill the missing values:", ["mean", "median", "mode", "constant"])

        constant_value = ""
        if fill_method == "constant":
            constant_value = st.text_input("Enter the constant value:")

        if st.button("Submit"):
            if action == "Fill missing values":
                if fill_method in ["mean", "median"] and not pd.api.types.is_numeric_dtype(st.session_state.data[column_to_handle]):
                    st.warning("Selected column is not numeric. Please choose another method or column.")
                else:
                    if fill_method == "mean":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(st.session_state.data[column_to_handle].mean())
                    elif fill_method == "median":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(st.session_state.data[column_to_handle].median())
                    elif fill_method == "mode":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(st.session_state.data[column_to_handle].mode()[0])
                    elif fill_method == "constant":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(constant_value)

                    st.success(f"Missing values in {column_to_handle} filled with {fill_method}!")
                    
            elif action == "Drop column":
                st.session_state.data.drop(columns=[column_to_handle], inplace=True)
                st.success(f"Column {column_to_handle} dropped!")

            elif action == "Drop rows with missing values":
                st.session_state.data.dropna(subset=[column_to_handle], inplace=True)
                st.success(f"Rows with missing values in {column_to_handle} dropped!")

            # Use experimental_rerun to refresh the app state
            st.experimental_rerun()

    else:
        st.success("There are no missing values in the dataset!")


def display_process_currency_percentage():
    # Detect columns with currency and percentage
    currency_cols = [col for col in st.session_state.data.columns if st.session_state.data[col].astype(str).str.contains('\$').any()]
    percent_cols = [col for col in st.session_state.data.columns if st.session_state.data[col].astype(str).str.contains('%').any()]

    # Initialize ignored columns in session state if not present
    if 'ignored_currency_cols' not in st.session_state:
        st.session_state.ignored_currency_cols = []
    if 'ignored_percent_cols' not in st.session_state:
        st.session_state.ignored_percent_cols = []

    # Filter out ignored columns from the main list
    currency_cols = [col for col in currency_cols if col not in st.session_state.ignored_currency_cols]
    percent_cols = [col for col in percent_cols if col not in st.session_state.ignored_percent_cols]

    col1, col2 = st.columns(2)

    if currency_cols:
        with col1:
            col1.write("Columns with currency values:")
            col1.write(currency_cols)

            # Allow user to select which currency column to process
            currency_column_to_process = col1.selectbox("Select a currency column to process:", currency_cols)
            currency_action = col1.radio("Choose an action:", ["Remove $", "Ignore"])

            if col1.button(f"Submit {currency_column_to_process}"):
                if currency_action == "Remove $":
                    st.session_state.data[currency_column_to_process] = st.session_state.data[currency_column_to_process].replace('[\$,]', '', regex=True).astype(float)
                    col1.success(f"Processed {currency_column_to_process} by removing $!")
                elif currency_action == "Ignore":
                    st.session_state.ignored_currency_cols.append(currency_column_to_process)

                # Use experimental_rerun to refresh the app state
                st.experimental_rerun()

    if percent_cols:
        with col1:
            col1.write("Columns with percentage values:")
            col1.write(percent_cols)

            # Allow user to select which percentage column to process
            percent_column_to_process = col1.selectbox("Select a percentage column to process:", percent_cols)
            percent_action = col1.radio("Choose an action for percentage column:", ["Convert % to fraction", "Ignore"])

            if col1.button(f"Submit {percent_column_to_process}", key=f"SubmitButton_{percent_column_to_process}"):
                if percent_action == "Convert % to fraction":
                    st.session_state.data[percent_column_to_process] = st.session_state.data[percent_column_to_process].str.rstrip('%').astype('float') / 100.0
                    col1.success(f"Processed {percent_column_to_process} by converting % to fraction!")
                elif percent_action == "Ignore":
                    st.session_state.ignored_percent_cols.append(percent_column_to_process)

                # Use experimental_rerun to refresh the app state
                st.experimental_rerun()

    with col2:
        col2.write("Ignored columns with currency values:")
        col2.write(st.session_state.ignored_currency_cols)
        col2.write("Ignored columns with percentage values:")
        col2.write(st.session_state.ignored_percent_cols)


def display_drop_columns():
    """Displays a grid of columns with checkboxes to allow users to drop selected columns."""
    st.write("Select the columns you wish to drop:")

    # Arrange columns in a grid
    num_columns = 4  # Define the number of columns for the grid
    col_chunks = [st.session_state.data.columns[i:i + num_columns] for i in range(0, len(st.session_state.data.columns), num_columns)]

    columns_to_drop = []
    for chunk in col_chunks:
        cols = st.columns(len(chunk))
        for i, col_name in enumerate(chunk):
            if cols[i].checkbox(col_name):
                columns_to_drop.append(col_name)

    # Button to submit the selected columns to drop
    if st.button("Drop Selected Columns"):
        st.session_state.data.drop(columns=columns_to_drop, inplace=True)
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}!")
        st.experimental_rerun()




def display_data_transformation():
    st.write("Choose a data transformation method:")

    transformation_choice = st.selectbox(
        "Select a transformation method:",
        ["Normalization", "Standardization", "Log Transformation"]
    )

    # Initialize feedback_message in session state if not present
    if 'feedback_message' not in st.session_state:
        st.session_state.feedback_message = ""

    if st.button("Submit"):
        # Reset feedback message
        st.session_state.feedback_message = ""

        # Select only numeric columns
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out columns with all NaN values or infinite values
        valid_cols = [col for col in numeric_cols if not st.session_state.data[col].isna().all()]
        valid_cols = [col for col in valid_cols if not np.isinf(st.session_state.data[col]).any()]

        try:
            if transformation_choice == "Normalization":
                scaler = StandardScaler()
                st.session_state.data[valid_cols] = scaler.fit_transform(st.session_state.data[valid_cols])
                st.session_state.feedback_message = "Normalization successful!"

            elif transformation_choice == "Standardization":
                scaler = MinMaxScaler()
                st.session_state.data[valid_cols] = scaler.fit_transform(st.session_state.data[valid_cols])
                st.session_state.feedback_message = "Standardization successful!"

            elif transformation_choice == "Log Transformation":
                # Adding a small constant to avoid log(0)
                st.session_state.data[valid_cols] = np.log(st.session_state.data[valid_cols] + 1)
                st.session_state.feedback_message = "Log Transformation successful!"

        except Exception as e:
            st.session_state.feedback_message = f"{transformation_choice} failed! Error: {e}"

    # Display feedback message below the submit button
    st.write(st.session_state.feedback_message)



def display_encode_categorical():
    # Display a preview of the dataset
    st.write("Data preview:")
    st.write(st.session_state.data.head())

    # Allow user to select a column to encode
    categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        st.warning("No categorical columns found in the dataset!")
        return
    
    column_to_encode = st.selectbox("Select a column to encode:", categorical_cols)
    
    # Display a preview of the selected column
    st.write(f"Preview of {column_to_encode}:")
    st.write(st.session_state.data[column_to_encode].head())

    # Allow user to choose encoding method
    encoding_method = st.selectbox("Choose an encoding method:", ["One-Hot Encoding", "Label Encoding"])
    
    if st.button("Encode"):
        if encoding_method == "One-Hot Encoding":
            # Split the comma-separated values to get a list of amenities
            split_data = st.session_state.data[column_to_encode].str.split(',').apply(lambda x: [i.strip() for i in x if i])

            # Use pd.get_dummies on these lists to one-hot encode the data
            dummies = pd.get_dummies(split_data.apply(pd.Series).stack()).sum(level=0)
            
            # Join the one-hot encoded columns to the original dataframe and drop the original column
            st.session_state.data = pd.concat([st.session_state.data, dummies], axis=1)
            st.session_state.data.drop(columns=[column_to_encode], inplace=True)
            st.success(f"{column_to_encode} encoded using One-Hot Encoding!")
            
        elif encoding_method == "Label Encoding":
            # Label encode the selected column
            le = LabelEncoder()
            st.session_state.data[column_to_encode] = le.fit_transform(st.session_state.data[column_to_encode])
            st.success(f"{column_to_encode} encoded using Label Encoding!")
        
        # Refresh the app state
        st.experimental_rerun()


def display_time_series_features():
    st.write("Time Series Feature Engineering")

    # Check if dataset contains any datetime columns
    datetime_cols = st.session_state.data.select_dtypes(include=[np.datetime64]).columns.tolist()

    if not datetime_cols:
        st.warning("No datetime columns found in the dataset!")
        return
    
    column_to_process = st.selectbox("Select a datetime column to process:", datetime_cols)
    
    # Display a preview of the selected column
    st.write(f"Preview of {column_to_process}:")
    st.write(st.session_state.data[column_to_process].head())

    # Allow user to choose the feature extraction method
    feature_method = st.selectbox(
        "Choose a time series feature extraction method:",
        ["Date Extraction", "Lag Features", "Rolling Window"]
    )

    if feature_method == "Date Extraction":
        # Extract day, month, year, day of the week
        if st.button("Extract Date Features"):
            st.session_state.data[f"{column_to_process}_day"] = st.session_state.data[column_to_process].dt.day
            st.session_state.data[f"{column_to_process}_month"] = st.session_state.data[column_to_process].dt.month
            st.session_state.data[f"{column_to_process}_year"] = st.session_state.data[column_to_process].dt.year
            st.session_state.data[f"{column_to_process}_weekday"] = st.session_state.data[column_to_process].dt.weekday
            st.success(f"Date features extracted from {column_to_process}!")
            st.experimental_rerun()

    elif feature_method == "Lag Features":
        lag_period = st.number_input("Enter lag period:", value=1, min_value=1)
        if st.button("Generate Lag Features"):
            st.session_state.data[f"{column_to_process}_lag{lag_period}"] = st.session_state.data[column_to_process].shift(lag_period)
            st.success(f"Lag features with period {lag_period} generated for {column_to_process}!")
            st.experimental_rerun()

    elif feature_method == "Rolling Window":
        window_size = st.number_input("Enter window size for rolling average:", value=3, min_value=1)
        if st.button("Generate Rolling Window Feature"):
            st.session_state.data[f"{column_to_process}_rolling_avg{window_size}"] = st.session_state.data[column_to_process].rolling(window=window_size).mean()
            st.success(f"Rolling window feature with window size {window_size} generated for {column_to_process}!")
            st.experimental_rerun()



def display_convert_to_datetime():
    st.write("Select Columns to Convert to Datetime Format")

    # Break down the columns into chunks for the 3x3 grid
    col_chunks = [st.session_state.data.columns[i:i + 3] for i in range(0, len(st.session_state.data.columns), 3)]
    
    # Use session state to store selected columns to convert
    if 'columns_to_convert_grid' not in st.session_state:
        st.session_state.columns_to_convert_grid = []

    # Display the 3x3 grid with checkboxes
    for chunk in col_chunks:
        cols = st.columns(3)
        for i, col_name in enumerate(chunk):
            if len(chunk) > i:  # Check if the column exists (for the last row which might not be full)
                if cols[i].checkbox(col_name, key=col_name, value=(col_name in st.session_state.columns_to_convert_grid)):
                    if col_name not in st.session_state.columns_to_convert_grid:
                        st.session_state.columns_to_convert_grid.append(col_name)
                else:
                    if col_name in st.session_state.columns_to_convert_grid:
                        st.session_state.columns_to_convert_grid.remove(col_name)

    # Convert button
    if st.button("Convert Selected to Datetime"):
        for col in st.session_state.columns_to_convert_grid:
            try:
                st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                st.success(f"{col} successfully converted to datetime format!")
            except Exception as e:
                st.error(f"Error converting {col} to datetime: {e}")
        # Clear the session state for selected columns to avoid carrying over selections
        st.session_state.columns_to_convert_grid = []
        # Refresh the app state
        st.experimental_rerun()


def display_data_preview():
    st.write("Data Preview:")
    st.write(st.session_state.data.head(10))

   
# This function will be added to the sub_page options in your main() function.

def display_results_page():
    st.write("Results")
    st.write("Data Preview:")
    st.write(st.session_state.data.head(10))

    target_col = st.selectbox("Select the target column", st.session_state.data.columns)

    if st.button("Train and Evaluate Models"):
        X_train, X_test, y_train, y_test = prepare_data(st.session_state.data, target_col)
        models = create_models()
        models = fit_models(X_train, y_train, models)
        results_df = evaluate_models(X_test, y_test, models)
           
        # Display model results in Streamlit
        st.write("Model Evaluation Results:")
        st.write(results_df)

        # Plot model performance
        st.write("Model Performance Comparison:")
        st.pyplot(plot_model_performance(results_df))
        

def main():
    st.title("Interactive Model Builder")

    # Sidebar for navigation
    page = st.sidebar.radio("Choose a page:", ["Upload Data", "Feature Engineering", "Evaluate the Model"])
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = None

    if page == "Upload Data":
        # Step 1: Upload CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            if st.session_state.data is None:  # If data is not already loaded into session
                st.session_state.data = pd.read_csv(uploaded_file)
        
                # Step 2: Convert to lowercase
                st.session_state.data = st.session_state.data.applymap(lambda s: s.lower() if type(s) == str else s)
    
        # Check if data is not None before displaying the preview
        if st.session_state.data is not None:
            st.write("Uploaded data preview:")
            st.write(st.session_state.data.head())

    elif page == "Feature Engineering":
        if st.session_state.data is not None:
            sub_page = st.sidebar.radio(
            "Choose a task:",
            ["Data Preview","Handle Missing Values", "Process Currency and Percentage", "Drop Columns", 
             "Data Transformation", "Encoding Categorical Variables", "Time Series Features", 
             ]
            )
            if sub_page == "Handle Missing Values":
                display_handle_missing_values()
            elif sub_page == "Process Currency and Percentage":
                display_process_currency_percentage()
            elif sub_page == "Drop Columns":
                display_drop_columns()
            elif sub_page == "Data Transformation":
                display_data_transformation()
            elif sub_page == "Encoding Categorical Variables":
                display_encode_categorical()
            elif sub_page == "Time Series Features":
                display_time_series_features()
            elif sub_page == "Convert to Datetime":
                display_convert_to_datetime()
            elif sub_page == "Data Preview":
                display_data_preview()
            elif page == "Evaluate the model":
                evaluate_model_page()

        else:
            st.warning("Please upload data first.")
    
    elif page == "Evaluate the Model":
        display_results_page()

if __name__ == "__main__":
    main()
