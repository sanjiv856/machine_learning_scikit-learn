# Titanic Survival Prediction

This project implements a machine learning pipeline to predict the survival of passengers on the Titanic using various classification algorithms. The project involves data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation.

## Project Structure
- `data/` - Directory containing the dataset (`train.csv` and `test.csv`).
- `python_scikit-learn_titanic.py` - Main script containing the code for loading data, preprocessing, training models, and generating submissions.

## Running the Code

### Feature Engineering & Preprocessing:
Feature engineering is applied to create new features such as Family_Size, Is_Alone, Title, Age_Group, etc.
Preprocessing pipelines are defined for numerical and categorical features.

### Model Training & Hyperparameter Tuning:

Several classifiers are trained and tuned using GridSearchCV, including:
- Random Forest
- Extra Trees
- XGBoost
- Decision Tree
- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors

Best models and their parameters are saved as .pkl files.

### Key Libraries Used
- pandas - Data manipulation and analysis.
- numpy - Numerical computations.
- matplotlib & seaborn - Data visualization.
- scikit-learn - Machine learning library for model building and evaluation.
- xgboost - Implementation of gradient boosting algorithm.

### Feature Engineering
The following features are engineered:
- Family_Size - Number of family members onboard.
- Is_Alone - Binary feature indicating if the passenger was alone.
- Title - Extracted from passenger names.
- Age_Group - Binned age groups.
- Ticket_Number, Ticket_Location - Extracted from ticket information.
- Cabin_Alphabet, Cabin_Recorded - Extracted from cabin information.

### Hyperparameter Tuning
Hyperparameters are tuned using GridSearchCV with cross-validation to find the best model configuration.

### Feature Importance
Feature importance is plotted for the top 20 features for each model.
