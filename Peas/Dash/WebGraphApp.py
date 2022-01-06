# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def UpdateSWDGraphData():
    engine = create_engine('postgresql://cflfcl_Rainshelter_SWC:o654UkI6iGNwhzHu@database.powerplant.pfr.co.nz/cflfcl_Rainshelter_SWC')
    SWD = pd.read_sql_table('SoilWaterDeficit', engine,index_col='index')
    return SWD


def UpdateCanopyData():
    engine = create_engine('postgresql://cflfcl_Rainshelter_SWC:o654UkI6iGNwhzHu@database.powerplant.pfr.co.nz/cflfcl_Rainshelter_SWC')
    Canopy = pd.read_sql_table('TempEP', engine,index_col=['Irrigation', 'Date'])
    return Canopy


# +
app = dash.Dash()   #initialising dash app

def Graphs():
    ## Soil Water Deficit Graph
    SWD = UpdateSWDGraphData()
    Canopy = UpdateCanopyData()
    
    SWDfig = go.Figure([go.Scatter(x =  SWD['date'], y = SWD['2D'],line = dict(dash='solid',color = 'orange', width = 4), name = '2 Day'),
                     go.Scatter(x =  SWD['date'], y = SWD['7D'],line = dict(dash='solid',color = 'green', width = 4), name = '7 Day'),
                    go.Scatter(x =  SWD['date'], y = SWD['14D'],line = dict(dash='solid',color = 'purple', width = 4), name = '14 Day'),
                     go.Scatter(x =  SWD['date'], y = SWD['21D'],line = dict(dash='dash',color = 'orange', width = 4), name = '21 Day'),
                    go.Scatter(x =  SWD['date'], y = SWD['MD'],line = dict(dash='dash',color = 'green', width = 4), name = 'Mid Drought'),
                     go.Scatter(x =  SWD['date'], y = SWD['LD'],line = dict(dash='dash',color = 'purple', width = 4), name = 'Late Drought')])
    SWDfig.update_layout(xaxis_title = 'Date', yaxis_title = 'Soil Water Deficit (mm)',
                       autosize=False, width=1000, height=700, margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="LightSteelBlue"
                      )
    
    ## Canopy Temperature Graphs
    fPARfig = go.Figure([go.Scatter(x =  Canopy.loc['2D','date'], y = Canopy.loc['2D','fPAR'],line = dict(dash='solid',color = 'orange', width = 4), name = '2 Day'),
                     go.Scatter(x =  Canopy.loc['7D','date'], y = Canopy.loc['7D','fPAR'],line = dict(dash='solid',color = 'green', width = 4), name = '7 Day'),
                    go.Scatter(x =  Canopy.loc['14D','date'], y = Canopy.loc['14D','fPAR'],line = dict(dash='solid',color = 'purple', width = 4), name = '14 Day'),
                     go.Scatter(x =  Canopy.loc['21D','date'], y = Canopy.loc['21D','fPAR'],line = dict(dash='dash',color = 'orange', width = 4), name = '21 Day'),
                    go.Scatter(x =  Canopy.loc['MD','date'], y = Canopy.loc['MD','fPAR'],line = dict(dash='dash',color = 'green', width = 4), name = 'Mid Drought'),
                     go.Scatter(x =  Canopy.loc['LD','date'], y = Canopy.loc['LD','fPAR'],line = dict(dash='dash',color = 'purple', width = 4), name = 'Late Drought')])
    fPARfig.update_layout( xaxis_title = 'Date', yaxis_title = 'Surface Temperature (oC)',
                       autosize=False, width=1000, height=700, margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="LightSteelBlue"
                      )
    
    Tsfig = go.Figure([go.Scatter(x =  Canopy.loc['2D','date'], y = Canopy.loc['2D','Ts'],line = dict(dash='solid',color = 'orange', width = 4), name = '2 Day'),
                     go.Scatter(x =  Canopy.loc['7D','date'], y = Canopy.loc['7D','Ts'],line = dict(dash='solid',color = 'green', width = 4), name = '7 Day'),
                    go.Scatter(x =  Canopy.loc['14D','date'], y = Canopy.loc['14D','Ts'],line = dict(dash='solid',color = 'purple', width = 4), name = '14 Day'),
                     go.Scatter(x =  Canopy.loc['21D','date'], y = Canopy.loc['21D','Ts'],line = dict(dash='dash',color = 'orange', width = 4), name = '21 Day'),
                    go.Scatter(x =  Canopy.loc['MD','date'], y = Canopy.loc['MD','Ts'],line = dict(dash='dash',color = 'green', width = 4), name = 'Mid Drought'),
                     go.Scatter(x =  Canopy.loc['LD','date'], y = Canopy.loc['LD','Ts'],line = dict(dash='dash',color = 'purple', width = 4), name = 'Late Drought'),
                    go.Scatter(x =  Canopy.loc['LD','date'], y = Canopy.loc['LD','Ta'],line = dict(dash='solid',color = 'black', width = 4), name = 'Air Temp')])
    Tsfig.update_layout( xaxis_title = 'Date', yaxis_title = 'Surface Temperature (oC)',
                       autosize=False, width=1000, height=700, margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="LightSteelBlue"
                      )
    
    Efig = go.Figure([go.Scatter(x =  Canopy.loc['2D','date'], y = Canopy.loc['2D','E'],line = dict(dash='solid',color = 'orange', width = 4), name = '2 Day'),
                     go.Scatter(x =  Canopy.loc['7D','date'], y = Canopy.loc['7D','E'],line = dict(dash='solid',color = 'green', width = 4), name = '7 Day'),
                    go.Scatter(x =  Canopy.loc['14D','date'], y = Canopy.loc['14D','E'],line = dict(dash='solid',color = 'purple', width = 4), name = '14 Day'),
                     go.Scatter(x =  Canopy.loc['21D','date'], y = Canopy.loc['21D','E'],line = dict(dash='dash',color = 'orange', width = 4), name = '21 Day'),
                    go.Scatter(x =  Canopy.loc['MD','date'], y = Canopy.loc['MD','E'],line = dict(dash='dash',color = 'green', width = 4), name = 'Mid Drought'),
                     go.Scatter(x =  Canopy.loc['LD','date'], y = Canopy.loc['LD','E'],line = dict(dash='dash',color = 'purple', width = 4), name = 'Late Drought')])
    Efig.update_layout(xaxis_title = 'Date', yaxis_title = 'Surface Temperature (oC)',
                       autosize=False, width=1000, height=700, margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="LightSteelBlue"
                      )
    
    Irrigs = Canopy.index.get_level_values(0).drop_duplicates()
    Durats = ['Last 2 days','Last 7 days','Last 14 days','Last 21 days']
    index = pd.MultiIndex.from_product([Irrigs,Durats],names=['Irrigation','Duration'])
    dayDurat = pd.DataFrame(index=Durats, data= [2.0,7.0,14.0,21.0],columns=['durat'])
    WaterUse = pd.DataFrame(index = index, data = [dayDurat.loc[x[1],'durat'] for x in index], columns = ['daysDurat'])

    Today = datetime.date.today()
    Yesterday = Today - datetime.timedelta(days=1)
    for i in WaterUse.index:
        startDate = Today - datetime.timedelta(days=WaterUse.loc[i,'daysDurat'])
        duratWaterUse = Canopy.loc[i[0],'E'].loc[startDate:Yesterday].sum()
        WaterUse.loc[i,'WaterUsed'] = np.round(duratWaterUse)
    WUfig = px.bar(WaterUse.reset_index(), x="Irrigation", y="WaterUsed", color='Irrigation',text="WaterUsed",pattern_shape='Irrigation',
             facet_col="Duration",
             color_discrete_sequence=['orange','green','purple','orange','green','purple'],
             pattern_shape_sequence=['.','.','.','x','x','x'],
             category_orders={"Duration": ['Last 2 days','Last 7 days','Last 14 days','Last 21 days']})
    WUfig.update_layout(yaxis_title = 'Water Used (mm)',
                       autosize=False, width=1000, height=700, margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="LightSteelBlue"
                      )
    #WUfig.update_traces(marker_color=['orange','green','purple','white','white','white'])
    return html.Div(id = 'parent', children = [html.H1(id = 'SWD', children = 'Soil Water Deficit (Rain shelter Peas 2021/22)', 
                                                       style = {'textAlign':'left','marginTop':40,'marginBottom':40}),        
                                               dcc.Graph(id = 'SWDgraph', figure = SWDfig),
                                               html.H1(id = 'fPAR', children = 'Green Cover (Rain shelter Peas 2021/22)', 
                                                       style = {'textAlign':'left','marginTop':40,'marginBottom':40}),        
                                               dcc.Graph(id = 'fPARgraph', figure = fPARfig),
                                               html.H1(id = 'Ts', children = 'Surface temperature (Rain shelter Peas 2021/22)', 
                                                       style = {'textAlign':'left','marginTop':40,'marginBottom':40}),        
                                               dcc.Graph(id = 'Tsgraph', figure = Tsfig),
                                               html.H1(id = 'E', children = 'Estimated Evaopration (Rain shelter Peas 2021/22)', 
                                                       style = {'textAlign':'left','marginTop':40,'marginBottom':40}),        
                                               dcc.Graph(id = 'Egraph', figure = Efig),
                                               html.H1(id = 'WU', children = 'Water Use Over set periods (Rain shelter Peas 2021/22)', 
                                                       style = {'textAlign':'left','marginTop':40,'marginBottom':40}),        
                                               dcc.Graph(id = 'WUgraph', figure = WUfig)
                                              ]
                   )
 
app.layout = Graphs
# -

if __name__ == '__main__': 
    app.run_server(host = '0.0.0.0', port = '3006', debug=True)

