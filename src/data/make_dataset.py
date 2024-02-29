import pandas as pd
import numpy as np  
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso, Ridge


def load_dataset(file_path):
    """
    Load the dataset from the specified file path and handle potential errors.
    Parameters:
    - file_path: str, path to the dataset file.
    Returns:
    - DataFrame: loaded dataset or None if an error occurs.
    """
    try:
        data_set = pd.read_csv(file_path)
        print(data_set.head())
        return data_set
    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        print("The file is empty. Please provide a valid dataset.")
    except Exception as e:
        print("An error occurred:", e)
    return None


def check_data_quality(data_set):
    """
    Check for duplicates, null, and missing values in the dataset.
    Parameters:
    - data_set: DataFrame, the dataset to be checked.
    Returns:
    - None, prints the findings.
    """
    if data_set is not None:
        # Check for duplicates
        duplicates = data_set.duplicated().sum()
        print(f'Number of duplicate rows: {duplicates}')
        # Check for null/missing values in each column
        null_values = data_set.isnull().sum()
        print('Null values in each column:')
        print(null_values)
    else:
        print("Data set is None, cannot perform data quality checks.")


def plot_categorical_distribution(df, column, top_n=20):
    """
    Function for the EDA step - for plotting categorical variables with too many
    categories (such as 'city' colmns etc)
    """
    # If there are too many categories, we'll take the top `top_n` and group the rest as 'Other'
    top_categories = df[column].value_counts().head(top_n).index
    df_top_categories = df.copy()
    df_top_categories[column] = df[column].where(df[column].isin(top_categories), 'Other')
    plt.figure(figsize=(15, 8))
    sns.countplot(data=df_top_categories, x=column, order=df_top_categories[column].value_counts().index)
    plt.title(f'Frequency of top {top_n} categories in {column}')
    plt.xticks(rotation=90)  
    plt.show()


### 1. Basic undersatning of the dataset
## Inspecion of the dataset
file_path = "../../data/raw/data.csv"
data_set = load_dataset(file_path)
check_data_quality(data_set)

## Additional inspection of the dataset
data_set.info()
print("Dataset columns:", data_set.columns)
data_set.describe()


### 2. Date Preproccessing 
# Convert the 'date' column to a datetime object
data_set['date'] = pd.to_datetime(data_set['date'])
# Verify the conversion by displaying the data types of each column
print(data_set.dtypes)


### 3. Exploratory Data Analysis (EDA)
# Calculate the Mutual Information (MI) scores between each of the
# independent variables and the continuous target variable 'price'.
# Copy the dataset for MI calculation to avoid altering the original dataset
mi_data_set = data_set.copy()
for column in ['date', 'street', 'city', 'statezip', 'country']:
    mi_data_set[column], _ = mi_data_set[column].factorize()
# Creating a boolean array for discrete features in the copied dataset
discrete_features_mi = [mi_data_set[col].dtype == 'int64' for col in mi_data_set.columns.drop('price')]
mi_scores = mutual_info_regression(mi_data_set.drop('price', axis=1), mi_data_set['price'], discrete_features=discrete_features_mi, random_state=0)
mi_scores_series = pd.Series(mi_scores, index=mi_data_set.columns.drop('price')).sort_values()
print(mi_scores_series)
plt.figure(figsize=(10, 8))
plt.barh(mi_scores_series.index, mi_scores_series.values)
plt.title('Mutual Information Scores')
plt.xlabel('MI Score')
plt.ylabel('Features')
plt.show()

# Checking the relationship between the continuous numerical
# variables and the target variable 'price' with pearson correlation and vizualisation.
continuous_vars = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'yr_built']
# Generate regression plots and print Pearson correlation values
for var in continuous_vars:
    correlation, p_value = stats.pearsonr(data_set[var], data_set['price'])
    print(f'Pearson correlation of {var} with Price: correlation={correlation}, p-value={p_value}')
    plt.figure(figsize=(8, 4))
    sns.regplot(x=var, y='price', data=data_set, line_kws={"color": "red"})
    plt.title(f'Regression plot of {var} vs. Price')
    plt.xlabel(var)
    plt.ylabel('Price')
    plt.show()

# Checking the relationship between the categorical variables
# and the target variable 'price' with ANOVA statistics and vizualisation.
categorical_vars = ['waterfront', 'view', 'condition']
# Generate box plots and compute ANOVA statistics
for var in categorical_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=var, y='price', data=data_set)
    plt.title(f'Box plot of Price vs. {var}')
    plt.show()
    groups = data_set.groupby(var)['price'].apply(list)
    f_value, p_value = stats.f_oneway(*groups)
    print(f'ANOVA statistic for {var} with Price: F-value={f_value}, p-value={p_value}')

# Checking the relationship between the discrete variables
# and the target variable 'price' with pearson correlation and vizualisation.
discrete_var = 'yr_renovated'
# Generate a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=discrete_var, y='price', data=data_set)
plt.title(f'Bar plot of Price vs. {discrete_var}')
plt.show()
correlation, p_value = stats.pearsonr(data_set[discrete_var], data_set['price'])
print(f'Pearson correlation of {discrete_var} with Price: correlation={correlation}, p-value={p_value}')
# Conclusion: We don't see a linear relationship between any of those
# variables and the target variable (and this is why the pearson
# correlation don't capture the relatioships as it capturs only linear ones),
# but we do see outliers in some of them. We'll handle them later once we make
# the features selection.

# First feature selection and dataset update based on the conclusion
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above'
                     ,'street', 'city', 'statezip', 'yr_built', 'price']
data_set = data_set[selected_features]
# Ensuring the updated dataset contains only the 'selected_features' 
print(data_set.info())

# Checking for multicollinearity among numerical independent variables by calculating VIF.
# Selecting only the numerical columns
numerical_vars = data_set.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_vars = [var for var in numerical_vars if var != 'price']  
# Check for constant or empty columns in numerical_vars
for var in numerical_vars:
    if data_set[var].nunique() <= 1:  # Column has 0 or 1 unique values
        print(f"Column {var} is constant or empty and will be excluded from VIF calculation.")
        numerical_vars.remove(var)
# VIF calculation if there are enough variables left
if len(numerical_vars) > 0:
    X = add_constant(data_set[numerical_vars])
    vif_data = pd.DataFrame({
        'Feature': X.columns.drop('const'),  
        'VIF': [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]  
    })
    print(vif_data)
else:
    print("Not enough variables for VIF calculation after excluding constant/empty columns.")

# Identifying outliers for continuous and discrete variables in the updated
# dataset - visually and statistically
continuous_discrete_vars = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'yr_built']
for var in continuous_discrete_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data_set[var])
    plt.title(f'Box plot of {var}')
    plt.show()
# Calculate IQR and identify outliers
for var in continuous_discrete_vars:
    Q1 = data_set[var].quantile(0.25)
    Q3 = data_set[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data_set[(data_set[var] < lower_bound) | (data_set[var] > upper_bound)]
    print(f'Number of outliers in {var} using IQR: {outliers.shape[0]}')
# Calculate Z-score and identify outliers
for var in continuous_discrete_vars:
    z_scores = np.abs(stats.zscore(data_set[var]))
    outliers = data_set[z_scores > 3]
    print(f'Number of outliers in {var} using Z-score: {outliers.shape[0]}')
# Check distribution for skewness
for var in continuous_discrete_vars:
    skewness = data_set[var].skew()
    plt.figure(figsize=(10, 6))
    sns.histplot(data_set[var], kde=True)
    plt.title(f'Distribution of {var} (Skewness: {data_set[var].skew():.2f})')
    plt.show()
    print(f'Skewness of {var}: {skewness}')
# Conclusion: All the continuous and discrete variables have outliers except of the
# 'yr_built' discrete variable.

# Identifying outliers for categorical variables in the updated dataset- visually 
categorical_vars = ['street', 'city', 'statezip']
# Set the number of categories I want to display
TOP_N_CATEGORIES = 20
# Generate bar plots for frequency visualization, and box plots for 'price' distribution across categories
for var in categorical_vars:
    plot_categorical_distribution(data_set, var)
    top_categories = data_set[var].value_counts().nlargest(TOP_N_CATEGORIES).index
    # Filter the data set for only the top N categories
    data_top_n = data_set[data_set[var].isin(top_categories)]
    plt.figure(figsize=(12, 8))  
    sns.boxplot(x='price', y=var, data=data_top_n, orient='h')  
    plt.title(f'Box plot of Price vs. {var} (Top {TOP_N_CATEGORIES} categories)')
    plt.show()

# Handling outliers using the Z-score method
# List of variables to check for outliers, excluding 'yr_built' as it has no outliers
variables_to_check = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above']
# Compute Z-scores of the dataset for the specified variables, and get absolute Z-scores
# to identify outliers
z_scores = stats.zscore(data_set[variables_to_check])
abs_z_scores = np.abs(z_scores)
# Filter the dataset: keep only the rows with all Z-scores less than 3
filtered_entries = (abs_z_scores < 3).all(axis=1)
dataset_cleaned = data_set[filtered_entries]
print(f"Original dataset shape: {data_set.shape}")
print(f"Cleaned dataset shape: {dataset_cleaned.shape}")
# Now, 'dataset_cleaned' is my working dataset 

# Encoding the independent categorical variables using 'MEstimateEncoder'
encoder = ce.MEstimateEncoder(cols=['city', 'street', 'statezip'], m=0.5)
# We don't drop 'price' here as we haven't split our data yet
dataset_encoded = encoder.fit_transform(dataset_cleaned.drop('price', axis=1), dataset_cleaned['price'])
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset_encoded, dataset_cleaned['price'],  
    test_size=0.2, random_state=2
)
print("Number of training samples:", X_train.shape[0])
print("Number of test samples:", X_test.shape[0])
# Normalize the feature matrix for the training and test sets
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
print("Shape of the normalized feature training matrix (X):", X_train_normalized.shape)
print("Shape of the normalized feature testing matrix (X):", X_test_normalized.shape)


### 4. Model Development 
# Initialize the models with default parameters, except where specified
models = {
    "Multiple Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=2),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=2),
    "Boosted Decision Tree Regression": GradientBoostingRegressor(random_state=2),
    "K-nearest neighbors (KNN)": KNeighborsRegressor(),
    "XGBoost": XGBRegressor(random_state=2),
    "ElasticNet": ElasticNet(random_state=2),
    "Lasso": Lasso(random_state=2),
    "Ridge": Ridge(random_state=2)
}
# Initial Model Training and Evaluation
initial_predictions = {}
metrics_df = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R² Score"])
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_normalized, y_train)
    preds = model.predict(X_test_normalized)
    initial_predictions[name] = preds
    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    # Create a new DataFrame for the current model's metrics
    new_metrics = pd.DataFrame({
        "Model": [name],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "R² Score": [r2]
    })
    # Concatenate the new metrics to the existing DataFrame
    metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)
    # Print evaluation results
    print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Visualization of Initial Evaluation Results
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i, metric in enumerate(["MAE", "MSE", "RMSE", "R² Score"]):
    sns.barplot(x='Model', y=metric, data=metrics_df, ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(f'Model Comparison - {metric}')
    axs[i//2, i%2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
# Identify the Best Model Based on Initial Evaluation
best_initial_model_name = metrics_df.loc[metrics_df['R² Score'].idxmax(), 'Model']
best_initial_r2 = metrics_df.loc[metrics_df['R² Score'].idxmax(), 'R² Score']
print(f"Best initial model based on R² Score: {best_initial_model_name} with R²: {best_initial_r2:.4f}")

# Hyper-parameter Tuning by applying cross-validation with GridSearchCV
if best_initial_model_name == "K-nearest neighbors (KNN)":
    print("Hyper-parameter Tuning for KNN...")
    param_grid = {
        'n_neighbors': range(1, 31),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10, scoring='r2')
    grid_search.fit(X_train_normalized, y_train)
    print(f"Best parameters for KNN: {grid_search.best_params_}")
    print(f"Best score for KNN: {grid_search.best_score_:.4f}")
    # Update KNN with the best parameters
    best_model = grid_search.best_estimator_
else:
    # For other models, I will add similar tuning processes or use the model as is
    best_model = models[best_initial_model_name]

# Final Model Selection and Evaluation
best_model.fit(X_train_normalized, y_train)  
final_predictions = best_model.predict(X_test_normalized)
# Final evaluation metrics
final_mae = mean_absolute_error(y_test, final_predictions)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(y_test, final_predictions)
print(f"Final Model ({best_initial_model_name}) Evaluation: MAE: {final_mae:.4f}, MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f}, R²: {final_r2:.4f}")
# Visualization of the Final Model's Performance
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs. Predicted - {best_initial_model_name}')
plt.show()