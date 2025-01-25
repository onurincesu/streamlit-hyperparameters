import streamlit as st
import pandas as pd
from models import XGBoost, LightGBM, DecisionTree, RandomForest, LinearRegressionModel, find_optimal_parameters

df=pd.read_csv("kc_house_data.csv")
df=df.drop(columns={"id","date","zipcode","yr_renovated"},axis=1)

with st.sidebar:
    selected_option = st.selectbox("Select an option", ["LightGBM", "XGBoost", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
    st.header("Model Settings")

    scaling_method = st.radio(
        "Select Scaling Method",
        ["None", "Standardize", "MinMax"],
        help="Choose how to scale the features"
    )
    
    split_size = st.slider("Split Size", 
                          min_value=0.1, 
                          max_value=0.9,
                          value=0.3, 
                          help="The split size determines the proportion of data used for testing (e.g., 0.3 means 30% test, 70% train)")

    use_optimal = st.checkbox("Use Optimal Parameters", value=False, help="Use grid search to find the best parameters")
    
    if use_optimal:
        y = df["price"]
        X = df.drop("price", axis=1)
        model_type_map = {
            "XGBoost": "XGBoost",
            "LightGBM": "LightGBM",
            "Decision Tree Regressor": "DecisionTree",
            "Random Forest Regressor": "RandomForest",
            "Linear Regression": "LinearRegression"
        }
        params = find_optimal_parameters(X, y, model_type_map[selected_option])

    if selected_option in ["XGBoost", "LightGBM"]:
        max_depth = st.slider("Max Depth", min_value=1, max_value=10, 
                            value=params['max_depth'] if use_optimal else 3)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, 
                                value=params['learning_rate'] if use_optimal else 0.01)
        n_estimators = st.slider("Number of Estimators", min_value=1, max_value=1000, 
                               value=params['n_estimators'] if use_optimal else 10)
    
    elif selected_option == "Decision Tree Regressor":
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, 
                            value=params['max_depth'] if use_optimal else 5)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, 
                                    value=params['min_samples_split'] if use_optimal else 2)
        min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, 
                                   value=params['min_samples_leaf'] if use_optimal else 1)

    elif selected_option == "Random Forest Regressor":
        n_estimators = st.slider("Number of Trees", min_value=1, max_value=200, 
                               value=params['n_estimators'] if use_optimal else 100)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, 
                            value=params['max_depth'] if use_optimal else 5)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, 
                                    value=params['min_samples_split'] if use_optimal else 2)

    elif selected_option == "Linear Regression":
        fit_intercept = st.checkbox("Fit Intercept", 
                                  value=params['fit_intercept'] if use_optimal else True,
                                  help="If True, the model will calculate the y-intercept (b in y=mx+b). If False, the model assumes it passes through origin (y=mx)")
        normalize = st.checkbox("Normalize", 
                              value=False,
                              help="If True, the regressors will be normalized before regression. This is different from feature scaling and is specific to Linear Regression")

    st.write("[Github](https://github.com/onurincesu)")
    st.write("[Linkedin](https://www.linkedin.com/in/ali-onur-incesu-04bb59218/)")

def model_seç():
    scaling = None if scaling_method == "None" else scaling_method.lower()
    
    if selected_option == "XGBoost":
        results = XGBoost(df, learning_rate, max_depth, n_estimators, split_size, scaling)
    elif selected_option == "LightGBM":
        results = LightGBM(df, learning_rate, max_depth, n_estimators, split_size, scaling)
    elif selected_option == "Decision Tree Regressor":
        results = DecisionTree(df, max_depth, min_samples_split, min_samples_leaf, split_size, scaling)
    elif selected_option == "Random Forest Regressor":
        results = RandomForest(df, n_estimators, max_depth, min_samples_split, split_size, scaling)
    elif selected_option == "Linear Regression":
        results = LinearRegressionModel(df, fit_intercept, normalize, split_size, scaling)

    st.line_chart(results[0])
    st.write("Model Performance Metrics:")
    for metric, value in results[1].items():
        st.write(f"{metric}: {value}")
    
    with st.expander("Performance Metrics Explained"):
        st.markdown("""
        ### Understanding the Metrics:
        
        * **MSE (Mean Squared Error)**
            * Measures the average squared difference between predicted and actual values
            * Lower values indicate better performance
            * More sensitive to outliers due to squaring
        
        * **RMSE (Root Mean Squared Error)**
            * Square root of MSE
            * Provides error in the same unit as the target variable (price)
            * Lower values indicate better performance
        
        * **MAE (Mean Absolute Error)**
            * Average absolute difference between predicted and actual values
            * Less sensitive to outliers compared to MSE
            * Lower values indicate better performance
        
        * **R² (R-squared)**
            * Represents the proportion of variance in the target that's predictable from the features
            * Ranges from 0 to 1 (or 0% to 100%)
            * Higher values indicate better fit (1.0 being perfect prediction)
        
        * **Mean % Error**
            * Average percentage difference between predicted and actual values
            * Gives a relative measure of error
            * Lower values indicate better performance
            * Easier to interpret as it's in percentage form
        """)

model_seç()
