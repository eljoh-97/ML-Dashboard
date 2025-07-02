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


# Create interactive dash-app and visualize the output of the model
# Load theme for all graphs
load_figure_template("superhero")
app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

app.layout = dbc.Container([
    html.Div([
        html.H1("Machine Learning Dashboard", className='text-center'),
        
        html.Div(
            children=[html.Div(style={'margin-bottom': '20px'})]
        ),
        
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div("Training Accuracy", style={'font-weight': 'bold', 'font-size': '20px'}),  
                        html.Div(f"{train_accuracy}%", style={'font-size': '32px', 'margin-top': '5px'}),   
                    ],
                    style={
                        'text-align': 'center',  
                        'border': '1px solid #ccc',  
                        'padding': '10px',       
                        'width': '350px',       
                        'margin': '10px auto',   
                        'border-radius': '5px',  
                        'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)',  
                    },
                ),
                html.Div(
                    children=[
                        html.Div("Test Accuracy", style={'font-weight': 'bold', 'font-size': '20px'}), 
                        html.Div(f"{test_accuracy} %", style={'font-size': '32px', 'margin-top': '5px'}),    
                    ],
                    style={
                        'text-align': 'center',  
                        'border': '1px solid #ccc',  
                        'padding': '10px',      
                        'width': '350px',       
                        'margin': '10px auto',   
                        'border-radius': '5px',  
                        'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)', 
                    },
                )
            ],
            style={
                'display': 'flex',          
                'justify-content': 'center',
                'gap': '20px',
                'margin-bottom': '20px',
            },
        ),
        html.Div(
            children=[html.Div("Classification Report", style={'font-weight': 'bold', 'font-size': '20px'})]
        ),
        
        dash_table.DataTable(
            data=df_report.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_report.columns],
            style_cell={'textAlign': 'left'},
            style_header={
                        'backgroundColor': 'rgba(0, 0, 0, 0)',
                        'fontWeight': 'bold',
                        },
            style_data={'backgroundColor': 'rgba(0, 0, 0, 0)'},
            style_as_list_view=True,
        ),
        
        html.Div(
            children=[html.Div(style={'margin-bottom': '50px'})]
        ),

        html.Div(
            children=[html.Div("Feature Dropdown", style={'font-weight': 'bold', 'font-size': '20px'})]
        ),
        # Feature dropdown
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in head_X_feature_name],
            value=head_X_feature_name[0],
            placeholder="Select a feature",
            multi=False,
            style={
                'font-weight': 'bold',
                'border': '1px solid #ccc',
                'border-radius': '5px',
                'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'
                } 
        ),
        
        html.Div(
            children=[html.Div(style={'margin-bottom': '20px'})]
        ),
        
        html.Div(
            children=[html.Div("Classification Dropdown", style={'font-weight': 'bold', 'font-size': '20px'})]
        ),
        
        # Class dropdown
        dcc.Dropdown(
            id='class-dropdown',
            options=[{'label': f"Classification {i}", 'value': i} for i in range(3)],  
            value=0,  
            placeholder="Select a class",
            multi=False,
            style={
                'font-weight': 'bold',
                'border': '1px solid #ccc',
                'border-radius': '5px',
                'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)',
                } 
        ),
        
        html.Div(
            children=[html.Div(style={'margin-bottom': '20px'})]
        ),
        
        # SHAP plots
        html.Div(
            children=[html.Div("SHAP Feature Importance - Top 10", style={'font-weight': 'bold', 'font-size': '20px', 'margin-bottom': '20px'})]
        ),
        dcc.Graph(id='shap-summary-plot'),
        html.Div(
            children=[html.Div("SHAP Dependence Value", style={'font-weight': 'bold', 'font-size': '20px'})]
        ),
        dcc.Graph(id='shap-dependence-plot'),
    ]),
])

# Callback for SHAP summary plot
@app.callback(
    Output('shap-summary-plot', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_summary_plot(selected_feature):
    
    fig = px.bar(
        shap_summary, 
        x='Feature', 
        y='Importance', 
        height=500
    )
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig


# Callback for SHAP dependence plot
@app.callback(
    Output('shap-dependence-plot', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('class-dropdown', 'value')]
)
def update_dependence_plot(selected_feature, selected_class):
    try:
        # Validate feature and class selection
        if selected_feature not in head_X_feature_name:
            return px.scatter(title="Invalid feature selected", height=400)

        if selected_class not in range(4):  # Ensure the selected class is valid
            return px.scatter(title="Invalid classification selected", height=400)
        
        # Extract SHAP values for the selected class
        shap_values_selected_class = shap_values[:, :, selected_class]
        
        # Generate dependence plot
        fig = px.scatter(
            x=X_test[selected_feature],
            y=shap_values_selected_class[:, head_X_feature_name.index(selected_feature)],
            color_discrete_map={'positive' : 'blue', 'negative' : 'red'},
            title=f"{selected_feature} Class: {selected_class}",
            labels={'x': selected_feature, 'y': 'SHAP Value'},
            height=500
        )
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            title={'text' : f"{selected_feature} Class: {selected_class}",
                   'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}
            )
        return fig

    except Exception as e:
        return px.scatter(title=f"Error generating plot: {str(e)}", height=400)

if __name__ == '__main__':
    app.run(debug=True)