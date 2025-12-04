import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Data Generation and Overview
# -----------------------------

# Parameters for synthetic dataset
num_samples = 500         # Total number of strategies
strategy_length = 64      # Length of the strategy bit string
possible_moves = ['C', 'D']  # 'C' for cooperate, 'D' for defect

# Generate a random strategy as a string of C's and D's
def generate_random_strategy(length):
    return ''.join(random.choice(possible_moves) for _ in range(length))

# Simulate performance for a given strategy.
# Here, performance is simulated as a function of the cooperation rate (optimal near 60%).
def simulate_performance(strategy):
    coop_rate = strategy.count('C') / len(strategy)
    noise = random.gauss(0, 5)  # Gaussian noise
    performance = 100 - abs(coop_rate - 0.6) * 100 + noise
    return performance

# Create dataset of strategies and their performance scores
strategies = [generate_random_strategy(strategy_length) for _ in range(num_samples)]
performances = [simulate_performance(s) for s in strategies]
df = pd.DataFrame({'strategy': strategies, 'performance': performances})

# Feature extraction: convert each strategy into numerical features
def extract_features(strategy):
    features = {}
    features['coop_rate'] = strategy.count('C') / len(strategy)
    features['defect_rate'] = strategy.count('D') / len(strategy)
    # Count number of transitions between consecutive moves
    features['transitions'] = sum(1 for i in range(1, len(strategy)) if strategy[i] != strategy[i-1])
    # Cooperation rate in the first 10 moves
    first_ten = strategy[:10]
    features['first_ten_coop'] = first_ten.count('C') / len(first_ten)
    return features

# Apply feature extraction on all strategies and combine with performance
features_list = [extract_features(s) for s in strategies]
features_df = pd.DataFrame(features_list)
data = pd.concat([features_df, df['performance']], axis=1)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())
print("\nFirst few rows:")
print(data.head())

# -----------------------------
# 2. Statistical Analysis of Numeric Features
# -----------------------------

# List of numeric features
numeric_features = data.select_dtypes(include=["number"]).columns
print("\nNumeric Features:", numeric_features.tolist())

# Calculate mean, median, standard deviation, minimum, and maximum for each numeric feature
for feature in numeric_features:
    print(f"\nFeature: {feature}")
    print(f"  Mean: {data[feature].mean()}")
    print(f"  Median: {data[feature].median()}")
    print(f"  Standard Deviation: {data[feature].std()}")
    print(f"  Minimum: {data[feature].min()}")
    print(f"  Maximum: {data[feature].max()}")

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# -----------------------------
# 3. Data Visualization
# -----------------------------

# Plot histograms for all numeric features
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    plt.hist(data[feature], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# If you wish to visualize the distribution of the non-numeric 'strategy' column (as counts of unique strategies)
# note that the strategies are high cardinality, so this might not be very informative:
# plt.figure(figsize=(12, 8))
# sns.countplot(data=data, x='strategy')
# plt.title('Distribution of Strategies')
# plt.xlabel('Strategy')
# plt.ylabel('Count')
# plt.xticks(rotation=90)
# plt.show()

# -----------------------------
# 4. Machine Learning: Model Training and Evaluation
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# For our synthetic data, we assume no missing values; otherwise, you could drop or fill them:
data_cleaned = data.dropna()

# Separate features (X) and target (y)
X = data_cleaned.drop(columns=['performance'])
y = data_cleaned['performance']

# One-hot encoding is not required here as all features are numeric.
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.25)

# Store model results in dictionaries
tree_scores = {}
forest_scores = {}
gbr_scores = {}

# Train Decision Tree Model
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
tree_scores = {
    "train_score": decision_tree.score(X_train, y_train),
    "test_score": decision_tree.score(X_test, y_test),
    "r2": r2_score(y_test, y_pred_tree)
}

# Train Random Forest Model with n_estimators=500
random_forest = RandomForestRegressor(n_estimators=500, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_forest = random_forest.predict(X_test)
forest_scores = {
    "train_score": random_forest.score(X_train, y_train),
    "test_score": random_forest.score(X_test, y_test),
    "r2": r2_score(y_test, y_pred_forest)
}

# Train Gradient Boosting Model with n_estimators=500
gbr = GradientBoostingRegressor(n_estimators=500, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
gbr_scores = {
    "train_score": gbr.score(X_train, y_train),
    "test_score": gbr.score(X_test, y_test),
    "r2": r2_score(y_test, y_pred_gbr)
}

# Print model results
print("\nDecision Tree Results:")
print(tree_scores)

print("\nRandom Forest Results:")
print(forest_scores)

print("\nGradient Boosting Results:")
print(gbr_scores)

# Determine the best model based on test score and r2
models = {
    "Decision Tree": tree_scores,
    "Random Forest": forest_scores,
    "Gradient Boosting": gbr_scores
}

best_model = None
best_test_score = float('-inf')

for model_name, scores in models.items():
    if scores["test_score"] > best_test_score:
        best_model = (model_name, scores)
        best_test_score = scores["test_score"]

print(f"\nBest Model: {best_model[0]} with scores: {best_model[1]}")
