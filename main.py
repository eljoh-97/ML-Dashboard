import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from project_ai import train_accuracy, test_accuracy, X_test
from project_ai import df_report, head_X_feature_name, shap_summary, shap_values

# Create interactive dash-app and visualize the output of the model
# Load theme for all graphs
# load_figure_template("superhero")

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