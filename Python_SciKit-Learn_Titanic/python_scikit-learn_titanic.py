import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. Load and Preprocess Data
# ----------------------------

# Set working directory (modify this path as needed)
# os.chdir("")

# Load datasets
train_df = pd.read_csv('data/titanic_kaggle/train.csv')
test_df = pd.read_csv('data/titanic_kaggle/test.csv')

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    
    # Family Size
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    
    # Is Alone
    df['Is_Alone'] = np.where(df['Family_Size'] == 1, 1, 0)
    
    # Extract Title from Name
    df['Title'] = df['Name'].apply(lambda name: name.split(",")[1].split(".")[0].strip())
    
    # Name Length
    df['Name_Length'] = df['Name'].apply(len)
    
    # Ticket Number
    df['Ticket_Number'] = df['Ticket'].apply(lambda x: x.split(' ')[-1] if x.split(' ')[-1].isdigit() else np.nan).astype(float)
    
    # Ticket Number Counts
    df['Ticket_Number_Counts'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # Ticket Location
    df['Ticket_Location'] = df['Ticket'].apply(lambda x: x.split(' ')[0] if len(x.split(' ')) > 1 else 'Blank')
    ticket_location_mapping = {
        'SOTON/O.Q.':'SOTON/OQ', 'C.A.':'CA', 'CA.':'CA',
        'SC/PARIS':'SC/Paris', 'S.C./PARIS':'SC/Paris',
        'A/4.':'A/4', 'A/5.':'A/5', 'A.5.':'A/5',
        'A./5.':'A/5', 'W./C.':'W/C', 'S.W./PP':'SW/PP'  
    }
    df['Ticket_Location'] = df['Ticket_Location'].replace(ticket_location_mapping)
    
    # Cabin Information
    df['Cabin_Alphabet'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'Missing')
    df['Cabin_Recorded'] = np.where(df['Cabin'].notna(), 1, 0)
    
    # Age Group Binning
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], 
                             labels=['Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior'])
    
    return df

# Apply feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

train_df.head()
test_df.head()

# Define features and target
features_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
X = train_df.drop(features_to_drop, axis=1)
y = train_df['Survived']
X_test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# ----------------------------
# 2. Define Preprocessing Pipeline
# ----------------------------

# Numerical features
numeric_features = ['Age', 'Fare', 'Ticket_Number', 'Ticket_Number_Counts', 'Name_Length', 'Family_Size', 'SibSp', 'Parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=1))
])

# Categorical features with one-hot-endocing
categorical_ohe_features = ['Sex', 'Embarked', 'Ticket_Location', 'Age_Group']
categorical_ohe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Set sparse_output to False
])

# Categorical features with ordinal encoding 
categorical_ordinal_features = ['Title', 'Cabin_Alphabet', 'Pclass']
categorical_ordianal = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Set sparse_output to False
])

# Binary features
binary_features = ['Is_Alone', 'Cabin_Recorded']
binary_transformer = 'passthrough'  # No transformation needed

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('catohe', categorical_ohe, categorical_ohe_features),
        ('catordinal', categorical_ordianal, categorical_ordinal_features),
        ('bin', binary_transformer, binary_features)
    ]
)

# ----------------------------
# 3. Define Models and Grid Parameters
# ----------------------------

models = {
    'RandomForest': RandomForestClassifier(random_state=1),
    'ExtraTrees': ExtraTreesClassifier(random_state=1),
    'XGBoost': XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'), 
    'DecisionTree': DecisionTreeClassifier(),    
    'LogisticRegression': LogisticRegression(solver='liblinear', n_jobs=-1),
    'GaussianNB': GaussianNB(),    
    'KNN': KNeighborsClassifier(n_jobs=-1),
    # 'SVC': SVC(), # its taking a long time to run
}

param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__criterion': ['gini', 'entropy']
    },
    'ExtraTrees': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__criterion': ['gini', 'entropy']
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    },
    'DecisionTree': {
        'classifier__max_depth': [10, 20, 30], 
        'classifier__min_samples_split': [5, 10, 15],
        'classifier__min_samples_leaf': [1, 2, 4], 
        'classifier__criterion': ['gini', 'entropy']
    },
    'KNN': {
        'classifier__n_neighbors': [5, 10, 15, 20, 25], 
        'classifier__p': [1, 2], 
        'classifier__weights': ['uniform', 'distance']
    },
    'SVC': {
        'classifier__C': [0.1, 1, 10], 
        'classifier__kernel': ['linear', 'rbf', 'poly'], 
        'classifier__degree': [2, 3, 4],
        'classifier__gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'classifier__penalty': ['l2'], 
        'classifier__C': [0.1, 1, 10]
    },
    'GaussianNB': {},  
}

# ----------------------------
# 4. Train Models with Grid Search and Save Models
# ----------------------------

best_models = {}
results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    
    grid_search.fit(X, y)
    
    best_models[model_name] = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    results.append({
        'Model': model_name,
        'Best Score': best_score,
        'Best Params': best_params
    })
    
    print(f"{model_name} Best Score: {best_score:.4f}")
    print(f"{model_name} Best Params: {best_params}")
    
    # Save the model to disk
    model_filename = f'{model_name}_model.pkl'
    joblib.dump(best_models[model_name], model_filename)
    print(f"{model_name} saved as {model_filename}")

# Create results dataframe
results_df = pd.DataFrame(results).sort_values(by='Best Score', ascending=False)

# Display results
print("\nGrid Search Results:")
print(results_df)

# ----------------------------
# 5. Generate Predictions and Save Submission Files
# ----------------------------

for model_name, model in best_models.items():
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # Save to CSV
    submission.to_csv(f'submission_{model_name}.csv', index=False)
    print(f"Submission file for {model_name} saved.")

# ----------------------------
# 6. Plot and Save Feature Importances (Top 20)
# ----------------------------

def plot_feature_importances(model, model_name, top_n=20):
    # Check if model has feature_importances_
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        # Create dataframe for importances
        feat_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort features by importance and take the top_n
        feat_importances = feat_importances.sort_values(by='Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_importances)
        plt.title(f'{model_name} Feature Importances (Top {top_n})')
        plt.tight_layout()

        # Save the plot
        plot_filename = f'{model_name}_feature_importances.png'
        plt.savefig(plot_filename)
        plt.show()
        print(f"Feature importance plot for {model_name} saved as {plot_filename}")
    else:
        print(f"{model_name} does not have feature_importances_")

# Plot and save feature importances for each model
for model_name, model in best_models.items():
    plot_feature_importances(model, model_name, top_n=20)


shutil.copy(f'submission_{results_df.Model[0]}.csv', 'submission.csv')