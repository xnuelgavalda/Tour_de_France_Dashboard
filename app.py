#--------------------------------------------------------------
# Project: Tour de France Historic Analysis
# Author: X. Nuel Gavaldà
# Dataset: https://mavenanalytics.io/challenges/maven-tour-de-france-challenge/25

#-------------------------------------------------------
# Imports
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import sys
from icecream import ic

# Dash imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objs as go
import base64

import pathlib

# Other imports
import certifi
import ssl
import geopy.geocoders

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx


# ----------------------------------------------------------------
# Initialize the application
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Color definitions
colors = {
    "background": "#ffff00",
    "plotbackground":"#FFFFFF",
    "marginplotbackground":"#F5F5DC",
    "text": "#000000"
}

# ------------------------------------
# Import preprocessed data
# ------------------------------------

# Download data
tdf_finishers_p = pd.read_csv('./data_preprocessing/tdf_finishers_p.csv')
tdf_finishers_p = pd.DataFrame(tdf_finishers_p)

#print(tdf_finishers_p.head())
#ic(tdf_finishers_p.head())
#sys.exit()

tdf_stages_p = pd.read_csv('./data_preprocessing/tdf_stages_p.csv')
tdf_stages_p = pd.DataFrame(tdf_stages_p)

tdf_tours_p = pd.read_csv('./data_preprocessing/tdf_tours_p.csv')
tdf_tours_p = pd.DataFrame(tdf_tours_p)

tdf_winners_avg_p = pd.read_csv('./data_preprocessing/tdf_winners_avg_p.csv')
tdf_winners_avg_p = pd.DataFrame(tdf_winners_avg_p)

winners_clas_more1_p = pd.read_csv('./data_preprocessing/winners_clas_more1_p.csv')
winners_clas_more1_p = pd.DataFrame(winners_clas_more1_p)

unique_cities_start_p = pd.read_csv('./data_preprocessing/unique_cities_start_p.csv')
unique_cities_start_p = pd.DataFrame(unique_cities_start_p)

unique_cities_finish_p = pd.read_csv('./data_preprocessing/unique_cities_finish_p.csv')
unique_cities_finish_p = pd.DataFrame(unique_cities_finish_p)

df_p = pd.read_csv('./data_preprocessing/df_p.csv')
df_p = pd.DataFrame(df_p)

df2_p = pd.read_csv('./data_preprocessing/df2_p.csv')
df2_p = pd.DataFrame(df2_p)

tdf_winners_p = pd.read_csv('./data_preprocessing/tdf_winners_p.csv')
tdf_winners_p = pd.DataFrame(tdf_winners_p)

# Dropouts trend
# Add trend
x = np.array(list(tdf_tours_p['Year']))
y = np.array(list(tdf_tours_p['% of Dropouts']))

slope, intercept = np.polyfit(x, y, 1)

trendline_y = slope * x + intercept

# Data for Pie charts: Figure 5 and 5_1
stages_nat_p = pd.read_csv('./data_preprocessing/stages_nat_p.csv')
df_type_p = pd.read_csv('./data_preprocessing/df_type_p.csv')

# -------------------------------------------------
# Plots
#--------------------------------------------------

# Figure1: Distance and Average Speed per Year
fig1 = make_subplots(specs=[[{"secondary_y":True}]])

fig1.add_trace(go.Scatter(x=tdf_tours_p['Year'],
                          y=tdf_tours_p['Total_Distance_km'],
                          name='Total Distance (x1000 km)',
                          mode='lines+markers',
                          text=tdf_winners_p['Rider'],
                          hoverinfo='x+y+text',
                          hovertemplate='Year: %{x}'
                                        '<br>Total Distance: %{y} km'
                                        '<br>Winner: %{text}'
                          ),secondary_y=False)

fig1.add_trace(go.Scatter(x=tdf_winners_avg_p['Year'],
                          y=tdf_winners_avg_p['Avg Speed (km/h)'],
                          name='Avg Speed (Km/h)',
                          mode='lines+markers',
                          text=tdf_winners_p['Rider'],
                          hoverinfo='x+y+text',
                          hovertemplate='Year: %{x}'
                                        '<br>Total Distance: %{y} km'
                                        '<br>Winner: %{text}'
                          ), secondary_y=True)

fig1.add_annotation(text='World War I',
                    xref="paper", yref="paper",
                    x=0.135, y=0.1,
                    showarrow=False,
                    textangle=-90)

fig1.add_annotation(text='World War II',
                    xref="paper", yref="paper",
                    x=0.325, y=0.1,
                    showarrow=False,
                    textangle=-90)

fig1.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   xaxis_title="Year",
                   xaxis_tickangle=0,
                   #title='Total Distance and Number of Stages per Year',
                   legend=dict(x=0.91, y=0.05,xanchor='auto', yanchor='auto', bgcolor="white",bordercolor="Black", borderwidth=1),
                   hovermode='closest'
                   #legend=dict(orientation="h", entrywidth=70, yanchor="bottom",y=1.02, xanchor="right", x=1)
                   )

# Set x and y-axes titles
fig1.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig1.update_yaxes(title_text="Average Speed (km/h)", secondary_y=True,
                  showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig1.update_yaxes(title_text="Distance (x1000 km)", secondary_y=False,
                  showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure 2: Top Winners
fig2 = px.bar(winners_clas_more1_p.sort_values(by='Wins'),
              y='Rider',
              x='Wins',
              orientation='h')

fig2.update_traces(hovertemplate='Rider: %{y}'
                                 '<br>Wins: %{x}',
                   marker={"color": "blue",
                           "opacity": 0.5,
                           "line": {"width": 0.5,
                                    "color": "blue"}})

fig2.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   barmode='group',
                   xaxis_title="Wins",
                   yaxis_title="Rider",
                   xaxis_tickangle=0,
                   #title='Participants per Year',
                   hovermode='closest')

fig2.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure3: Participants and Dropouts per Year
fig3 = make_subplots(specs=[[{"secondary_y":True}]])

fig3.add_trace(go.Scatter(x=tdf_tours_p['Year'],
                          y=tdf_tours_p['Starters'],
                          name='Starters',
                          mode='lines+markers',
                          hovertemplate='Year: %{x}'
                                        '<br>Starters: %{y}'
                          ),secondary_y=False)

fig3.add_trace(go.Scatter(x=tdf_tours_p['Year'],
                          y=tdf_tours_p['Finishers'],
                          name='Finishers',
                          mode='lines+markers',
                          hovertemplate='Year: %{x}'
                                        '<br>Finishers: %{y}'
                          ), secondary_y=False)

fig3.add_trace(go.Scatter(x=tdf_tours_p['Year'],
                          y=tdf_tours_p['% of Dropouts'],
                          name='% of Dropouts',
                          mode='lines+markers',
                          hovertemplate='Year: %{x}'
                                        '<br>% Dropouts: %{y:.2f} %'
                          ), secondary_y=True)

trendline_trace = go.Scatter(x=x,
                             y=trendline_y,
                             mode='lines',
                             name='% Dropout Trendline',
                             hovertemplate='Year: %{x}'
                                           '<br>% Dropouts Trend: %{y:.2f} %')

fig3.add_trace(trendline_trace)

fig3.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   barmode='group',
                   xaxis_title="Year",
                   yaxis_title="Participants",
                   xaxis_tickangle=0,
                   #title='Participants per Year',
                   hovermode='closest',
                   legend=dict(x=0.51,
                               y=1.2,
                               xanchor='auto',
                               yanchor='auto',
                               bgcolor="white",
                               bordercolor="Black",
                               borderwidth=1))

fig3.add_annotation(text='World War I',
                    xref="paper", yref="paper",
                    x=0.14, y=0.1,
                    showarrow=False,
                    textangle=-90)

fig3.add_annotation(text='World War II',
                    xref="paper", yref="paper",
                    x=0.325, y=0.1,
                    showarrow=False,
                    textangle=-90)

fig3.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig3.update_yaxes(title_text="Participants", secondary_y=False,
                  showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig3.update_yaxes(title_text="Dropouts (%)", secondary_y=True,
                  showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure 3_1: Stage Distance per Year
fig3_1 = px.bar(df_p,
                x='Year',
                y='Stage',
                color='Type')

fig3_1.update_traces(hovertemplate='Year: %{x}'
                                   '<br>Stages: %{y}')

fig3_1.update_layout(plot_bgcolor=colors['plotbackground'],
                     paper_bgcolor=colors['marginplotbackground'],
                     font_color=colors['text'],
                     xaxis_tickangle=0,
                     xaxis_title="Winner",
                     yaxis_title="Number of Stages")

fig3_1.add_annotation(text='World War I',
                      xref="paper", yref="paper",
                      x=0.1, y=0.1,
                      showarrow=False,
                      textangle=-90)

fig3_1.add_annotation(text='World War II',
                      xref="paper", yref="paper",
                      x=0.32, y=0.1,
                      showarrow=False,
                      textangle=-90)

fig3_1.update_xaxes(showline=True, linewidth=1, linecolor='black',
                    showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig3_1.update_yaxes(showline=True, linewidth=1, linecolor='black',
                    showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure4: Average Speed
fig4 = px.scatter(tdf_winners_avg_p,
                  x='Rider',
                  y="Avg Speed (km/h)",
                  color='Country',
                  custom_data=['Country'])

fig4.update_traces(hovertemplate='Rider: %{x}'
                                 '<br>Avg. Speed: %{y} km/h'
                                 '<br>Country: %{customdata[0]}')

fig4.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   barmode='group',
                   xaxis_title="Rider",
                   yaxis_title="Average Speed (km/h)",
                   xaxis_tickangle=45,
                   hovermode='closest')

fig4.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig4.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure5: Nationality coutns of stage winners
unique_tdf_stages_p = tdf_stages_p['Winner_Nationality'].unique()

fig5 = px.pie(stages_nat_p,
              values='Count',
              names='Winner_Nationality',
              hole= 0.5)

fig5.update_traces(textposition='inside',
                   textinfo='label+percent',
                   hoverinfo='label+percent+value',
                   hovertemplate='Country: %{label}'
                                 '<br>Count: %{value}'
                                 '<br>Count (%): %{percent}')

fig5.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   showlegend=False)

# Figure5_1: Stage type counts
fig5_1 = px.pie(df_type_p,
                values='Count',
                names='Type',
                hole=0.5)

fig5_1.update_traces(textposition='inside',
                     insidetextorientation='horizontal',
                     textinfo='label+percent',
                     hoverinfo='label+percent+value',
                     hovertemplate='Country: %{label}'
                                   '<br>Count: %{value}'
                                   '<br>Count (%): %{percent}')


fig5_1.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   showlegend=False)

# Figure6: Number of stages per winner
fig6 = px.bar(df_p,
              x='Year',
              y='Stage',
              color='Type',
              title='Number of Stages by Winner',
              labels={'Year': 'Winner', 'Stage': 'Number of Stages'})

fig6.update_layout(plot_bgcolor=colors['background'],
                   paper_bgcolor=colors['background'],
                   font_color=colors['text'],
                   xaxis_tickangle=45,
                   xaxis_title="Winner",
                   yaxis_title="Number of Stages")

fig6.add_annotation(text='World War I',
                    xref="paper", yref="paper",
                    x=0.1, y=0.1,
                    showarrow=False,
                    textangle=-90)

fig6.add_annotation(text='World War II',
                    xref="paper",
                    yref="paper",
                    x=0.32,
                    y=0.1,
                    showarrow=False,
                    textangle=-90)

# Start and finish locations map
fig7 = px.scatter_mapbox(unique_cities_start_p,
                        lat='Latitude',
                        lon='Longitude',
                        color="Count",
                        size="Count",
                        hover_data=["City"],
                        #center = {'lat': 46.603354, 'lon': 1.888334},  # Centered around France
                        color_discrete_sequence=["City","Count"],
                        size_max=25,
                        zoom=4)

fig7.update_layout(mapbox_style="open-street-map")
fig7.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig8 = px.scatter_mapbox(unique_cities_finish_p,
                        lat='Latitude',
                        lon='Longitude',
                        color="Count",
                        size="Count",
                        hover_data=["City"],
                        #center = {'lat': 46.603354, 'lon': 1.888334},  # Centered around France
                        color_discrete_sequence=["City","Count"],
                        size_max=25,
                        zoom=4)

fig8.update_layout(mapbox_style="open-street-map")
fig8.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig8.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'])

# Finish location histogram
number_cities = 20
fig9 = px.bar(unique_cities_start_p.head(number_cities),
              x='City',
              y='Count')

fig9.update_layout(plot_bgcolor=colors['plotbackground'],
                   paper_bgcolor=colors['marginplotbackground'],
                   font_color=colors['text'],
                   xaxis_tickangle=45,
                   xaxis_title="City",
                   yaxis_title="Count",
                   title_text="Top Start Locations",
                   title_x =0.5)

fig9.update_traces(marker={"color":"green",
                            "opacity":0.5,
                            "line": {"width":0.5,
                                     "color":"green"}})

fig9.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig9.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

fig10 = px.bar(unique_cities_finish_p.head(number_cities),
              x='City',
              y='Count')

fig10.update_traces(marker={"color":"red",
                            "opacity":0.5,
                            "line": {"width":1,
                                     "color":"red"}})

fig10.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=45,
                    xaxis_title="City",
                    yaxis_title="Count",
                    title_text="Top Finish Locations",
                    title_x=0.5)

fig10.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig10.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Number of Stages won by the winners
fig11 = px.bar(df2_p,
               x='Winner_Name',
               y='Count',
               color='Winner_Nationality')

fig11.update_traces(hovertemplate='Country: %{x}'
                                  '<br>Count: %{y}')

fig11.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=45,
                    xaxis_title="Winner",
                    yaxis_title="Number of Stages")

fig11.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig11.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Stage distances over years
fig12 = px.scatter(df_p,
                   x='Year',
                   y='Distance_km',
                   color='Type',
                   custom_data=['Type'])

fig12.update_traces(hovertemplate='Year: %{x}'
                                  '<br>Distance: %{y} Km'
                                  '<br>Type: %{customdata[0]}')

fig12.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=0,
                    xaxis_title="Year",
                    yaxis_title="Distance (km)")

fig12.add_annotation(text='World War I',
                     xref="paper", yref="paper",
                     x=0.14, y=0.1,
                     showarrow=False,
                     textangle=-90)

fig12.add_annotation(text='World War II',
                     xref="paper",
                     yref="paper",
                     x=0.35, y=0.1,
                     showarrow=False,
                     textangle=-90)

fig12.update_xaxes(showline=True, linewidth=1, linecolor='black',
                   showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig12.update_yaxes(showline=True, linewidth=1, linecolor='black',
                   showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure 13: Horizontal histogram
#fig13 = go.Figure(data=[go.Histogram(y=tdf_stages_p['Distance_km'])])
fig13 = px.histogram(tdf_stages_p,
                     y="Distance_km",
                     marginal="box",
                     color_discrete_sequence=["blue"])

fig13.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=0,
                    xaxis_title="Count",
                    yaxis_title="Distance")

fig13.update_traces(marker={"color":"blue",
                            "opacity":0.5,
                            "line": {"width":1,
                                     "color":"blue"}},
                    hovertemplate='Count: %{x}'
                                  '<br>Distance: %{y}')

fig13.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig13.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure 14: Stage distances statistics


fig14 = px.box(tdf_stages_p,
             x='Year',
             y='Distance_km')

fig14.update_layout(plot_bgcolor=colors['background'],
                  paper_bgcolor=colors['background'],
                  font_color=colors['text'],
                  xaxis_tickangle=45,
                  xaxis_title="Year",
                  yaxis_title="Distance (km)")

fig14.add_annotation(text='World War I',
                  xref="paper", yref="paper",
                  x=0.14, y=0.1,
                  showarrow=False,
                  textangle=-90
                  )

fig14.add_annotation(text='World War II',
                  xref="paper", yref="paper",
                  x=0.35, y=0.1,
                  showarrow=False,
                  textangle=-90
                  )

fig14.update_traces(hovertemplate='Year: %{x}'
                                  '<br>Distance: %{y}')

fig14.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=0,
                    xaxis_title="Year",
                    yaxis_title="Stage Distance (km)")

fig14.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig14.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Figure 15:
fig15 = px.box(df_p,
               x='Type',
               y='Distance_km')

fig15.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=45,
                    xaxis_title="Stage Type",
                    yaxis_title="Distance (km)")

fig15.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig15.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

# Height and weight stats
tdf_winners_height = tdf_winners_p[tdf_winners_p['Height m'] != 0]  # remove zeros

fig16 = px.histogram(tdf_winners_height,
                     x="Height m",
                     marginal="box",
                     color_discrete_sequence=["orange"])

fig16.update_traces(marker={"color":"orange",
                            "opacity":0.5,
                            "line": {"width":2,
                                     "color":"orange"}})

fig16.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=0,
                    xaxis_title="Height (m)",
                    yaxis_title="Count")

fig16.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig16.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

tdf_winners_weight = tdf_winners_p[tdf_winners_p['Weight kg'] != 0]  # remove zeros

fig17 = px.histogram(tdf_winners_weight,
                     x="Weight kg",
                     marginal="box",
                     color_discrete_sequence=["red"])

fig17.update_traces(marker={"color":"red",
                            "opacity":0.5,
                            "line": {"width":1,
                                     "color":"red"}})

fig17.update_layout(plot_bgcolor=colors['plotbackground'],
                    paper_bgcolor=colors['marginplotbackground'],
                    font_color=colors['text'],
                    xaxis_tickangle=0,
                    xaxis_title="Weight (Kg)",
                    yaxis_title="Count")

fig17.update_xaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
fig17.update_yaxes(showline=True, linewidth=1, linecolor='black',
                  showgrid=True, gridwidth=1, gridcolor='#DCDCDC')


#---------------------------------------------
# Web layout
#---------------------------------------------

image_path = 'images/TdF_logo2.png'
encoded_image = base64.b64encode(open(image_path, 'rb').read())

app.layout = html.Div(
    style={
        'backgroundColor': colors['background']
    },
    children=[
        # Add the TdF.jpeg image in the left upper corner
        html.Div(
            [
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                         style={'width': '150px',
                                'float': 'left'
                                }
                         ),

        # Title and subtitle
        html.H1(
            children='Tour de France Historic Analysis',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'marginLeft': '5px'
            }
        ),

        html.Div([
            html.A(
                "Xavier Nuel Gavaldà",
                href="https://portfolio-xaviernuelgavalda-148e3f5b58a0.herokuapp.com/",
                target="_blank"
            ),
            html.Div(
                [
                    html.A("Dataset",
                           href="https://mavenanalytics.io/challenges/maven-tour-de-france-challenge/25",
                           target="_blank")
                ], style={'textAlign': 'center'}
            ),
        ]),

        html.H6(
            children=["The Tour de France is an annual men's multiple-stage bicycle race primarily held in France.\n"
                      "It is the oldest of the three Grand Tours (the Tour, the Giro d'Italia, "
                      "and the Vuelta a España) and is generally considered the most prestigious."
                      ],
            style={'textAlign': 'center',
                    'color': colors['text'],
                    'marginLeft': '35px',
                    'marginRight': '35px'
                }
        ),

        html.H6(
            children=["The race was first organized in 1903 to increase sales for the newspaper L'Auto and has been held annually since,"
                      " except when stopped for the two World Wars. As the Tour gained prominence and popularity the race was lengthened and "
                      "gained more international participation. The Tour is a UCI World Tour event, which means that the teams that compete "
                      "in the race are mostly UCI WorldTeams, with the exception of the teams that the organizers invite."
                      ],
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'marginLeft': '35px',
                'marginRight': '35px'
                }
        ),
    ],
            style={'textAlign': 'center'}
    ),

    html.Br(),

    # First column of plots
    html.Div(
        [
            html.H3(
                'Distance and Average Speed per Year',
                style={'textAlign': 'center'}),
            dcc.Graph(
                id='total-distance',
                config={'displayModeBar': False},
                figure=fig1)
        ],
        style={'width': '65%',
               'display': 'inline-block',
               'marginLeft': '20px'}
    ),

    html.Div(
        [
        html.H3(
            'Top Winners',
            style={'textAlign': 'center'}),
            dcc.Graph(
                id='winners-list',
                config={'displayModeBar': False},
                figure=fig2)
            ],
            style={'width': '30%',
                  'display': 'inline-block',
                  'margin_right': '20px',
                  "margin": 20,
                  "maxWidth": 800}
        ),

        html.Br(),
        html.Br(),

        html.Div(
            children=[
                html.H3(
                    children='Participants and Dropouts per Year'),
                    dcc.Graph(id='participants',
                              figure=fig3),
            ],
            style={'width': '80%',
                   'margin-left': '150px',
                   'textAlign': 'center'}
        ),

        html.Br(),
        html.Br(),

        html.Div(
            children=[
                html.Br(),
                html.Label(
                    'Stage Type',
                    style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='selector4',
                    options=[{'label':category, 'value':category} for category in df_p['Type'].unique()],
                    placeholder='Select a category',
                    style={'textAlign': 'center',
                           'marginLeft': '280px',
                           'marginRight': '850px'}
                ),

                html.Div(
                    children=[
                        html.H3(
                            children='Stages Distance over Years'),
                        dcc.Graph(
                            id='stages_histogram2',
                            figure=fig12),
                    ],
                    style={'width': '62%',
                           'display': 'inline-block',
                           'margin-left': '30px',
                           'textAlign': 'center'}
                ),

                html.Div(
                    children=[
                        html.H3(
                            children='Stages Distance over Years'),
                    dcc.Graph(id='stages_histogram3',
                                  figure=fig13),
                    ],
                    style={'width': '30%',
                          'display': 'inline-block',
                          'margin-left': '30px',
                          'textAlign': 'center'}
                ),
            ]
        ),

        html.Br(),
        html.Br(),

        html.Div(
            [
                html.H3(children='Stage Type Counts',
                        style={'textAlign': 'center'}),
            dcc.Graph(
                id='pie',
                config={'displayModeBar': False},
                figure=fig5_1)
            ],
            style={'width': '30%',
                  'display': 'inline-block',
                  'margin-left': '35px',
                  'textAlign': 'center'}
        ),

        html.Div(
            children=[
                html.H3(children='Stage Type Statistics'),
                dcc.Graph(id="stat2",
                          figure=fig15)
            ],
            style={'width': '63%',
                   'display': 'inline-block',
                   'margin-left': '35px',
                   'textAlign': 'center'}
        ),

        html.Br(),
        html.Br(),
        html.Br(),

        html.Div(
            children=[
                html.H3(children='Number and Type of Stages per Year'),
                html.Label('Stage Type', style={'textAlign': 'center'}),
                dcc.Dropdown(id='selector3',
                             options=[{'label':category, 'value':category} for category in df_p['Type'].unique()],
                             placeholder='Select a category',
                             style={'textAlign': 'center',
                                    'marginLeft': '200px',
                                    'marginRight': '600px'}
                ),
                html.Br(),
                dcc.Graph(id='participants3_3',
                          figure=fig3_1),
            ],
            style={'width': '80%',
                  'margin-left': '150px',
                  'textAlign': 'center'}
        ),

        html.Br(),
        html.Br(),

        html.Div(
            [
                html.H3(
                    children='Nationality Counts of Stage Winners',
                    style={'textAlign': 'center'}),
                dcc.Graph(
                    id='pie2',
                    config={'displayModeBar': False},
                    figure=fig5)
            ],
            style={'width': '30%',
                  'display': 'inline-block',
                  'margin-left': '35px',
                  'textAlign': 'center'}
        ),

        html.Div(
            children=[
                html.H3(
                    children='Number of Stages Won by TdF Winners'),
                dcc.Graph(id='stages_histogram11',
                          figure=fig11),
            ],
            style={'width': '63%',
                   'display': 'inline-block',
                   'margin-left': '35px',
                   'textAlign': 'center'}
        ),

        html.Br(),
        html.Br(),

        html.Div(
            [
                html.H3(
                    children='Average Speed per TdF Winners and Year',
                    style={'textAlign': 'center'}),
                html.Label(
                    'Country',
                    style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='selector1',
                    options=[{'label':category, 'value':category} for category in tdf_winners_avg_p['Country'].unique()],
                    placeholder='Select a category',
                    style={'textAlign': 'center',
                           'marginLeft': '200px',
                           'marginRight': '600px'}
                ),
                html.Br(),
                dcc.Graph(
                    id='average-speed',
                    figure=fig4)
        ],
            style={'width': '80%',
                  'margin-left': '150px',
                  'textAlign': 'center'}
        ),

        html.Div(
            [
                html.H4(
                    'Height TdF Winners Statistics',
                    style={'textAlign': 'center'}),
                dcc.Graph(
                    id='stat_height',
                    config={'displayModeBar': False},
                    figure=fig16)
            ],
            style={'width': '45%',
                   'display': 'inline-block',
                   'marginLeft': '60px'}
        ),

        html.Div(
            [
                html.H4(
                    'Weight TdF Winners Statistics',
                    style={'textAlign': 'center'}),
            dcc.Graph(
                id='stat_weight',
                config={'displayModeBar': False},
                figure=fig17)
            ],
            style={'width': '45%',
                   'display': 'inline-block',
                   'margin_right': '20px',
                   "margin": 20, "maxWidth": 800}
        ),

        html.Br(),
        html.Br(),

        html.H3(
            "Start and Finish Locations Map Distribution",
            style={"font-size": "30px",
                   "textAlign": "center"},
        ),

        html.Label(
            'Location Category',
            style={'textAlign': 'center'}
        ),

        dcc.RadioItems(
            id='selector2',
            options=[{'label': 'Start','value': 'Start'},
                     {'label': 'Finish', 'value': 'Finish'}],
            value='Start',
            style={'textAlign': 'center',
                   'border': '1px solid black',
                   'marginLeft': '680px',  # Add left margin
                   'marginRight': '680px'  # Add right margin
                   }
        ),

        html.Div(
            [
            dcc.Graph(
                id="map1",
                figure=fig7,
                style={'display': 'inline-block'}),
            ],
            style={'width': '48%',
                  'display': 'inline-block',
                  'marginLeft': '20px'}
        ),

        html.Div(
            [
            dcc.Graph(
                id="bar1",
                figure=fig9,
                style={'display': 'inline-block'}),
        ],
            style={'width': '30%',
                   'display': 'inline-block',
                   "margin": 20,
                   'margin_right': '40px',
                   "maxWidth": 800}
        ),
    ]
)

# Callback for Start and Finish Locations Distribution
@app.callback(
    [Output("map1", "figure"),
     Output("bar1", "figure")],
    [Input("selector2", "value")]
)
def update_figures(selected_value):
    if selected_value == 'Start':
        map_figure = fig7
        bar_figure = fig9
    elif selected_value == 'Finish':
        map_figure = fig8
        bar_figure = fig10
    else:
        map_figure = fig7
        bar_figure = fig9

    return map_figure, bar_figure


# Callback to update the output based on Country dropdown selection
@app.callback(
     Output('average-speed', 'figure'),
    [Input('selector1', 'value')]
)
def update_output(selected_country):
    if selected_country:
        filtered_tdf_winners_avg_p = tdf_winners_avg_p[tdf_winners_avg_p['Country'] == selected_country]
        # Create a scatter plot for average speed per winners and year
        new_fig = px.scatter(filtered_tdf_winners_avg_p,
                  x='Rider',
                  y="Avg Speed (km/h)",
                  color='Country',
                  custom_data=['Country'])

        new_fig.update_layout(plot_bgcolor=colors['plotbackground'],
                            paper_bgcolor=colors['marginplotbackground'],
                            font_color=colors['text'],
                            xaxis_title="Rider",
                            yaxis_title="Average Speed (km/h)",
                            xaxis_tickangle=45,
                            hovermode='y unified')

        new_fig.update_xaxes(showline=True, linewidth=1, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
        new_fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

        return new_fig

    # If no country is selected, return empty figures
    else:
        return fig4

# Callback to update the output based on Stage type dropdown selection
@app.callback(
     Output('participants3_3', 'figure'),
    [Input('selector3', 'value')]
)
def update_output(selected_type):
    if selected_type:
        filtered_df_p = df_p[df_p['Type'] == selected_type]
        # Create a scatter plot with filtered data
        new_fig = px.bar(filtered_df_p,
                       x='Year',
                       y='Stage',
                       color='Type')

        new_fig.update_traces(hovertemplate='Year: %{x}'
                                          '<br>Stage: %{y}')

        new_fig.update_layout(plot_bgcolor=colors['plotbackground'],
                             paper_bgcolor=colors['marginplotbackground'],
                             font_color=colors['text'],
                             xaxis_tickangle=0,
                             xaxis_title="Winner",
                             yaxis_title="Number of Stages")


        new_fig.update_xaxes(showline=True, linewidth=1, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
        new_fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

        return new_fig

    # If no country is selected, return empty figures
    else:
        return fig3_1


# Callback for Figure12 and Figure13
@app.callback(
    [Output("stages_histogram2", "figure"),
     Output("stages_histogram3", "figure")],
    [Input("selector4", "value")]
)
def update_output(selected_type):
    if selected_type:
        # Update scatter plot
        filtered_df_p = df_p[df_p['Type'] == selected_type]
        new_scat = px.scatter(filtered_df_p,
                           x='Year',
                           y='Distance_km',
                           color='Type',
                           custom_data=['Type'])

        new_scat.update_traces(hovertemplate='Year: %{x}'
                                          '<br>Distance: %{y} Km'
                                          '<br>Type: %{customdata[0]}')

        new_scat.update_layout(plot_bgcolor=colors['plotbackground'],
                            paper_bgcolor=colors['marginplotbackground'],
                            font_color=colors['text'],
                            xaxis_tickangle=0,
                            xaxis_title="Year",
                            yaxis_title="Distance (km)")

        new_scat.update_xaxes(showline=True, linewidth=1, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
        new_scat.update_yaxes(showline=True, linewidth=1, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

        # Update histogram
        filtered_tdf_stages_p = tdf_stages_p[tdf_stages_p['Type'] == selected_type]

        new_hist = px.histogram(filtered_tdf_stages_p,
                             y="Distance_km",
                             marginal="box",
                             color_discrete_sequence=["blue"])

        new_hist.update_layout(plot_bgcolor=colors['plotbackground'],
                        paper_bgcolor=colors['marginplotbackground'],
                        font_color=colors['text'],
                        xaxis_tickangle=0,
                        xaxis_title="Count",
                        yaxis_title="Distance")

        new_hist.update_traces(marker={"color": "blue",
                                    "opacity": 0.5,
                                    "line": {"width": 1,
                                             "color": "blue"}},
                           hovertemplate='Count: %{x}'
                                          '<br>Distance: %{y}')

        new_hist.update_xaxes(showline=True, linewidth=1, linecolor='black',
                       showgrid=True, gridwidth=1, gridcolor='#DCDCDC')
        new_hist.update_yaxes(showline=True, linewidth=1, linecolor='black',
                       showgrid=True, gridwidth=1, gridcolor='#DCDCDC')

        return new_scat, new_hist

    # If no country is selected, return empty figures
    else:
        return fig12, fig13



if __name__ == '__main__':
    app.run(debug=True)