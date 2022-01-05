# sqlalchemy and pandas solution 
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.pool import NullPool
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import datetime
import pandas as pd
import numpy as np


app = dash.Dash()   #initialising dash app

def SWDGraph():
    SWD = get_data()
    # SWD.set_index("date", inplace=True)
    # Function for creating line chart showing Google stock prices over time 
    fig = go.Figure([go.Scatter(x =  SWD['date'], y = SWD['2D'],line = dict(dash='solid',color = 'orange', width = 4), name = '2 Day'),
                     go.Scatter(x =  SWD['date'], y = SWD['7D'],line = dict(dash='solid',color = 'green', width = 4), name = '7 Day'),
                    go.Scatter(x =  SWD['date'], y = SWD['14D'],line = dict(dash='solid',color = 'purple', width = 4), name = '14 Day'),
                     go.Scatter(x =  SWD['date'], y = SWD['21D'],line = dict(dash='dash',color = 'orange', width = 4), name = '21 Day'),
                    go.Scatter(x =  SWD['date'], y = SWD['MD'],line = dict(dash='dash',color = 'green', width = 4), name = 'Mid Drought'),
                     go.Scatter(x =  SWD['date'], y = SWD['LD'],line = dict(dash='dash',color = 'purple', width = 4), name = 'Late Drought')])
    fig.update_layout(title = 'Soil Water Deficit (mm)',
                      xaxis_title = 'Date',
                      yaxis_title = 'Soil Water Deficit (mm)',
                       autosize=False,
                      width=1000,
                      height=700,
                      margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                        ),
                      paper_bgcolor="LightSteelBlue"
                      )
    return html.Div(id = 'parent', children = [
           html.H1(id = 'H1', children = 'Rain shelter Peas 2021/22', style = {'textAlign':'center','marginTop':40,'marginBottom':40}),        
           dcc.Graph(id = 'line_plot', figure = fig)])
 
app.layout = SWDGraph

if __name__ == '__main__': 
    app.run_server(host = '0.0.0.0', port = '3006', debug=True)

#https://stackoverflow.com/questions/51170169/clean-up-database-connection-with-sqlalchemy-in-pandas
def get_data():
    query = 'SELECT * FROM "RainShelterPea2022";'
    try:
        engine = create_engine("postgresql://cflfcl_Rainshelter_SWC:o654UkI6iGNwhzHu@database.powerplant.pfr.co.nz/cflfcl_Rainshelter_SWC", 
        poolclass=NullPool)
        df = pd.read_sql_query(query, engine)
    finally:
        engine.dispose()
    return df