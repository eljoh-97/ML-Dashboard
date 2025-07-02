import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.metrics import accuracy_score, classification_report
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


# Load data
current_dir  = Path(__file__).parent    
df_file = current_dir / "sum_customer_data.csv"
df_file = pd.read_csv(df_file)

# Preprocess X, y
X = df_file.drop(columns=['customerClassification'], errors='ignore')
y = LabelEncoder().fit_transform(df_file['customerClassification'])

# Define the dtypes and feature groups
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Split into categorical features to low / mod / high for cardinality of the dataset
low_cat_features = [i for i in categorical_features if X[i].nunique() <= 10]
mod_cat_features = [i for i in categorical_features if 10 < X[i].nunique() <= 50]
high_cat_features = [i for i in categorical_features if X[i].nunique() > 50]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  
        ('low_card_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_cat_features), 
        ('mod_card_cat', LeaveOneOutEncoder(), mod_cat_features),
        ('high_card_cat', TargetEncoder(), high_cat_features) 
    ]
)

# Encode target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data - training and test 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#Fit X_train & y_train with preprocessor
X_train_processed = preprocessor.fit_transform(X_train, y_train)
#Fit X test with preprocessor
X_test_processed = preprocessor.transform(X_test)

# Get feature names 
low_cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(low_cat_features)
numerical_feature_names = numerical_features
mod_cat_feature_names = mod_cat_features 
high_cat_feature_names = high_cat_features

X_feature_names = (list(numerical_feature_names) + 
                 list(low_cat_feature_names) + 
                 list(mod_cat_feature_names) + 
                 list(high_cat_feature_names))


# Train model
rf = RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=42)
rf.fit(X_train_processed, y_train)

# Evaluate & predict the data
train_accuracy = round(accuracy_score(y_train, rf.predict(X_train_processed)) * 100, 1)
test_accuracy = round(accuracy_score(y_test, rf.predict(X_test_processed)) * 100, 2)

# Create classification report & create a df output
report = classification_report(y_test, rf.predict(X_test_processed), output_dict=True)
df_report = pd.DataFrame(report).transpose().round(2)
df_report.insert(0, 'Measures', ['Class 0', 'Class 1', 'Class 3', 
                          'accuracy', 'macro avg', 'weighted avg'])


# Feature Importance
rf_feature_importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature' : X_feature_names, 'Importance' : rf_feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Define SHAP Values
t_explainer = shap.TreeExplainer(rf)
shap_values = t_explainer.shap_values(X_test_processed)

shap_summary = pd.DataFrame({
    'Feature': X_feature_names,
    'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

# Filer by top 10, reflect the dependence plot based on shap_summary
shap_summary = shap_summary.head(10)
head_X_feature_name = shap_summary.drop(columns=('Importance')).reset_index(drop=True)
head_X_feature_name = head_X_feature_name.values.flatten().tolist()