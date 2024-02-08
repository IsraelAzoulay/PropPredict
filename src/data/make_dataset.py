import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_dataset(file_path):
    """
    Load the dataset from the specified file path and handle potential errors.
    Parameters:
    - file_path: str, path to the dataset file.
    Returns:
    - DataFrame: loaded dataset or None if an error occurs.
    """
    try:
        # Reading the CSV file into a DataFrame
        data_set = pd.read_csv(file_path)
        # Display the first few rows of the DataFrame to verify it's loaded correctly
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


def plot_price_distribution(data_set, column='price', log_transform=False):
    """
    Plot the distribution of the target column, optionally applying a logarithmic transformation.
    Parameters:
    - data_set: DataFrame, the dataset containing the target column.
    - column: str, the name of the target column to plot. Default is 'price'.
    - log_transform: bool, whether to apply a logarithmic transformation to the target column before plotting.
    """
    # Apply logarithmic transformation if specified
    if log_transform:
        # Applying log transformation for price
        data_set['Logprice'] = np.log1p(data_set['price'])
        target_column = 'Logprice'
        title_suffix = ' (Log Transformed)'
    else:
        target_column = column
        title_suffix = ''

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data_set[target_column], kde=True)
    plt.title(f'Distribution of {column}{title_suffix}')
    plt.xlabel(f'{column}{title_suffix}')
    plt.ylabel('Frequency')
    plt.show()

    # Plotting the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data_set[target_column])
    plt.title(f'Box Plot of {column}{title_suffix}')
    plt.xlabel(f'{column}{title_suffix}')
    plt.show()


def evaluate_skewness(data_set):
    """
    Evaluate the skewness of numerical features in the dataset.
    Parameters:
    - data_set: DataFrame, the dataset to be evaluated.
    Returns:
    - None, prints the skewness values for each numerical feature.
    """
    skewness = data_set.select_dtypes(include=['int64', 'float64']).skew().sort_values(ascending=False)
    print("Skewness of numerical features:")
    print(skewness)


def plot_feature_distributions(data_set):
    """
    Plot histograms for numerical features in the dataset to visually inspect their distributions.
    Parameters:
    - data_set: DataFrame, the dataset containing the features.
    """
    num_features = data_set.select_dtypes(include=['int64', 'float64']).columns
    data_set[num_features].hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.tight_layout()
    plt.show()


def encode_categorical_columns(data_set):
    # Function to encode categorical columns in the dataset
    for col in data_set.select_dtypes(include=['object', 'category']).columns:
        data_set[col], _ = data_set[col].factorize()
    return data_set


def calculate_mutual_info_scores(data_set, target_column, discrete_feature_names):
    # Function to calculate mutual information scores
    data_set_encoded = encode_categorical_columns(data_set.copy())
    
    # Convert the names of the discrete features to indices
    discrete_features_indices = [data_set_encoded.columns.get_loc(name) for name in discrete_feature_names]

    # Compute the mutual information scores with the target
    mutual_info_scores = mutual_info_regression(
        data_set_encoded.drop(target_column, axis=1), 
        data_set_encoded[target_column], 
        discrete_features=discrete_features_indices
    )
    
    # Create a Series with the scores and columns
    mutual_info_scores_series = pd.Series(mutual_info_scores, index=data_set_encoded.drop(target_column, axis=1).columns)

    # Return the scores sorted in descending order
    return mutual_info_scores_series.sort_values(ascending=False)


def remove_outliers_z_score_threshold(data_set, threshold=3):
    """
    Remove outliers from the dataset using Z-score method with a customizable threshold.
    Parameters:
    - data_set: DataFrame, the dataset to remove outliers from.
    - threshold: int, the threshold value for Z-score. Default is 3.
    Returns:
    - DataFrame: dataset with outliers removed.
    """
    # Selecting only numerical columns for outlier removal
    numerical_columns = data_set.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculating Z-score for each numerical column
    z_scores = data_set[numerical_columns].apply(lambda x: np.abs((x - x.mean()) / x.std()))
    
    # Removing rows where any Z-score is greater than the threshold
    data_set_cleaned = data_set[(z_scores < threshold).all(axis=1)]
    
    return data_set_cleaned


def target_encode(data_set, feature, target):
    """
    Target encode a categorical feature in the dataset based on the target variable.
    Parameters:
    - data_set: DataFrame, the dataset containing the feature to be encoded.
    - feature: str, the name of the categorical feature to encode.
    - target: str, the name of the target variable.
    Returns:
    - DataFrame: dataset with the categorical feature encoded.
    """
    if feature not in data_set.columns:
        raise KeyError(f"The feature '{feature}' is not found in the DataFrame.")
    
    data_set_copy = data_set.copy()  # Create a copy of the DataFrame
    target_mean = data_set_copy.groupby(feature)[target].mean()
    encoded_feature_name = f'{feature}_encoded'
    data_set_copy.loc[:, encoded_feature_name] = data_set_copy[feature].map(target_mean)
    data_set_copy.drop(columns=[feature], inplace=True)  # Remove the original feature
    return data_set_copy


def train_evaluate_compare_algorithms(data_set, target_column='price'):
    # Splitting the data into features and target
    X = data_set.drop(columns=[target_column])
    y = data_set[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regression': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regression': GradientBoostingRegressor(random_state=42),
        'Support Vector Regression': SVR(),
        'XGBoost Regression': XGBRegressor(random_state=42)
    }

    # Training and evaluating models
    scores = {}
    best_model_name = None
    best_cross_val_score = float('-inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_cross_val_score = np.mean(cross_val_scores)
        scores[name] = {
            'Mean Squared Error': mse, 
            'Mean Absolute Error': mae, 
            'R-squared': r2,
            'Mean Cross-Validated R-squared': mean_cross_val_score
        }

        # Update best model based on cross-validation
        if mean_cross_val_score > best_cross_val_score:
            best_cross_val_score = mean_cross_val_score
            best_model_name = name

    # Displaying scores
    for name, score in scores.items():
        print(f"Scores for {name}:")
        for metric, value in score.items():
            print(f"{metric}: {value}")
        print("-----------------------")

    print(f"The best model based on cross-validation is: {best_model_name}")
    return scores




## Inspecion of the dataset
file_path = "../../data/raw/data.csv"
data_set = load_dataset(file_path)
check_data_quality(data_set)

## Additional inspection of the dataset
data_set.info()
print("Dataset columns:", data_set.columns)

# Manually specify which int64 features are truly discrete based on your knowledge of the dataset
true_discrete_features = ['bedrooms', 'bathrooms', 'floors', 'view', 'condition']

# Plotting the distribution of the 'price' column
plot_price_distribution(data_set, column='price')

# Evaluate skewness & distributions of numerical features
evaluate_skewness(data_set)
plot_feature_distributions(data_set)

# Calculate mutual information scores for the 'price' column
mutual_info_scores = calculate_mutual_info_scores(data_set, 'price', true_discrete_features)
print(mutual_info_scores)
# Plot the mutual information scores
plt.figure(figsize=(12, 8))
mutual_info_scores.sort_values().plot(kind='barh')
plt.title('Mutual Information Scores')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.show()

# Selecting the top features based on mutual information scores
# (I update it based on the new mutual information scores plot)
selected_features = ['price', 'street','statezip','city','sqft_living','sqft_above',
                     'bathrooms','yr_built','sqft_lot','bedrooms']

# Creating the refined dataset based on selected features
refined_data_set = data_set[selected_features]

## Target encoding for 'statezip' and 'city' features
refined_data_set_encoded = target_encode(refined_data_set, 'statezip', 'price')
refined_data_set_encoded = target_encode(refined_data_set_encoded, 'city', 'price')
refined_data_set_encoded = target_encode(refined_data_set_encoded, 'street', 'price')

# Removing outliers from the refined dataset using Z-score method.
refined_data_set_encoded_cleaned = remove_outliers_z_score_threshold(refined_data_set_encoded)

# Train, evaluate, and compare algorithms
train_evaluate_compare_algorithms(refined_data_set_encoded_cleaned, 'price')













