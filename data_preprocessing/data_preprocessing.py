# ------------------------------------
# Preprocess data
# ------------------------------------

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import pickle
import re
import sys
from icecream import ic

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import certifi
import ssl
import geopy.geocoders
from geopy.geocoders import Nominatim
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

#--------------------------------------------------------------
# Figure1
# Download tours data
tdf_tours = pd.read_csv("../data/tdf_tours.csv", decimal=",",encoding="ISO-8859-1")

# Convert object to strings
tdf_tours = tdf_tours.convert_dtypes()

# Replace the string '+' to ','
tdf_tours['Stages'] = tdf_tours['Stages'].apply(lambda x: x.replace(' + ', ', '))
tdf_tours[['Stages_Number','Stages_Information','col']] = tdf_tours['Stages'].str.split(', ', expand=True)
tdf_tours['Stages_Information'] = tdf_tours['Stages_Information'].fillna('No Data')
tdf_tours[['Total_Distance_km', 'Total_Distance_mi']] = tdf_tours['Distance'].str.split('km', expand=True)

# replace ',' to '.'
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].str.replace(',','.')
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].str.strip()

# There is a string to
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].str.replace('3.660.5','3.660')
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].str.replace('3.360.3','3.360')
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].str.replace('3.414.4','3.414')
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].str.strip()
tdf_tours['Total_Distance_km'] = tdf_tours['Total_Distance_km'].astype(float)

tdf_tours['% of Dropouts'] = ((tdf_tours['Starters'] - tdf_tours['Finishers']) / tdf_tours['Starters'])*100

df_tours = tdf_tours[tdf_tours['% of Dropouts'] > 0]
df_tours = pd.DataFrame(df_tours)


# Save preprocessed data
tdf_tours.to_csv('tdf_tours_p.csv', index=False)

# --------------------------------------------------------------
# Download finishers data
tdf_finishers = pd.read_csv("../data/tdf_finishers.csv", decimal=",")

# Fill null data mit categorical data
tdf_finishers['Time'] = tdf_finishers['Time'].fillna('No Data')
tdf_finishers['Team'] = tdf_finishers['Team'].fillna('No Data')

# Save the dataset
tdf_finishers.to_csv('tdf_finishers_p.csv', index=False)

# -------------------------------------------------------------------
# Download stages data
tdf_stages = pd.read_csv("../data/tdf_stages.csv", decimal=",")

# Let's split the Course, Distance and Winner columns
tdf_stages[['Start','Finish','to']] = tdf_stages['Course'].str.split('to', expand=True)
tdf_stages = tdf_stages.drop(['to'],axis=1)
tdf_stages[['Distance_km', 'Distance_mi']] = tdf_stages['Distance'].str.split(' ', expand=True)
tdf_stages[['Winner_Name','Winner_Nationality','test']]= tdf_stages['Winner'].str.split('(', expand=True)
tdf_stages[['Winner_Nationality','test']]= tdf_stages['Winner_Nationality'].str.split(')', expand=True)
tdf_stages = tdf_stages.drop(['test'], axis=1)

tdf_stages['Distance_km'] = tdf_stages['Distance_km'].apply(lambda x: x.replace('km',''))
tdf_stages['Distance_km'] = tdf_stages['Distance_km'].astype(float)

# Save dataset
tdf_stages.to_csv('tdf_stages_p.csv', index=False)

# Data for pie of Figure5: Stages won by nationality
stages_nat = tdf_stages.groupby('Winner_Nationality', as_index=False)["Stage"].count().rename(columns={"Stage":"Count"})
stages_nat.to_csv('stages_nat_p.csv', index=False)


#---------------------------------------------------------------------
# Winners data
tdf_winners = pd.read_csv("../data/tdf_winners.csv", encoding="ISO-8859-1")

tdf_winners['Time'] = tdf_winners['Time'].fillna('0')
tdf_winners['Margin'] = tdf_winners['Margin'].fillna('0')
tdf_winners['Avg Speed'] = tdf_winners['Avg Speed'].fillna('0')
tdf_winners['Height'] = tdf_winners['Height'].fillna('0')
tdf_winners['Weight'] = tdf_winners['Weight'].fillna('0')

# top winners list
winners_clas = tdf_winners.groupby(['Rider'])['Year'].count().sort_values(ascending=False)
winners_clas = winners_clas.reset_index(name='Wins')
winners_clas_more1 = winners_clas[winners_clas['Wins'] >= 2]
winners_clas_more1 = pd.DataFrame(winners_clas_more1)
winners_clas_more1.to_csv('winners_clas_more1_p.csv', index=True)

# Average Speed by Years
tdf_winners['Avg Speed'] = tdf_winners['Avg Speed'].apply(lambda x: x.replace('km/h', ''))
tdf_winners['Height'] = tdf_winners['Height'].apply(lambda x: x.replace('m', ''))
tdf_winners['Weight'] = tdf_winners['Weight'].apply(lambda x: x.replace('kg', ''))
tdf_winners.rename(columns={"Avg Speed":"Avg Speed (km/h)"}, inplace=True)
tdf_winners.rename(columns={"Height":"Height m"}, inplace=True)
tdf_winners.rename(columns={"Weight":"Weight kg"}, inplace=True)

tdf_winners.to_csv('tdf_winners_p.csv', index=True)

tdf_winners_avg = tdf_winners[tdf_winners['Avg Speed (km/h)'] != '0']  # remove zeros
tdf_winners_avg['Avg Speed (km/h)'] = tdf_winners_avg['Avg Speed (km/h)'].astype(float)
tdf_winners_avg.to_csv('tdf_winners_avg_p.csv', index=True)

# Height and weight stats



#----------------------------------------------------------
# Stages preprocessing
df = tdf_stages.groupby(['Year','Type','Distance_km'])['Stage'].count()
df = df.reset_index()

df['Type'] = df['Type'].apply(lambda x: x.replace('Stage with mountain', 'Stage with mountains'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Stage with mountain(s)', 'Stage with mountains'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Stage with mountains(s)', 'Stage with mountains'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Stage with mountainss', 'Stage with mountains'))

df['Type'] = df['Type'].apply(lambda x: x.replace('Mountain Stage', 'Mountain stage'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Mountain Stage (s)', 'Mountain stage'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Mountain stage (s)', 'Mountain stage'))

df['Type'] = df['Type'].apply(lambda x: x.replace('Hilly Stage', 'Hilly stage'))

df['Type'] = df['Type'].apply(lambda x: x.replace('Medium-mountain stage', 'Medium mountain stage'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Medium mountain stage[c]', 'Medium mountain stage'))

df['Type'] = df['Type'].apply(lambda x: x.replace('Flat stage cobblestone stage', 'Flat stage with cobblestones'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Plain stage with cobblestone', 'Flat stage with cobblestones'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Flat cobblestone stage', 'Flat stage with cobblestones'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Flat stage with cobblestoness', 'Flat stage with cobblestones'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Flat stage with cobblestones', 'Stage with cobblestones'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Flat Stage', 'Flat stage'))

df['Type'] = df['Type'].apply(lambda x: x.replace('Plain stage', 'Flat stage'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Flat stage stage', 'Flat stage'))
df['Type'] = df['Type'].apply(lambda x: x.replace('Flat stage', 'Flat'))

# Save dataset
df.to_csv('df_p.csv', index=True)

# Dataset for pie of Figure 5_1:
df_type = df.groupby("Type", as_index=False)["Stage"].count().rename(columns={"Stage":"Count"})
df_type.to_csv('df_type_p.csv', index=False)


#-------------------------------------------------------------
# Figure 11
df2 = tdf_stages.groupby(['Winner_Name', 'Winner_Nationality'], as_index=False)['Stage'].count().rename(columns={"Stage":"Count"}).sort_values(by='Count', ascending=False)
df2 = df2.head(50)  # first 40 winners
df2 = df2.reset_index()
df2.to_csv('df2_p.csv', index=True)



sys.exit()

#-------------------------------------------------------------
# Count the number of times each city appears as a start or finish stage
city_counts_start = tdf_stages['Start'].value_counts()

# Get unique city names with counts
unique_cities_start = city_counts_start.reset_index()
unique_cities_start.columns = ['City', 'Count']

city_counts_finish = tdf_stages['Finish'].value_counts()

# Get unique city names with counts
unique_cities_finish = city_counts_finish.reset_index()
unique_cities_finish.columns = ['City', 'Count']

# Convertion
def convert_coord(city=''):
    geolocator = Nominatim(user_agent="MyApp",scheme='http')
    location = geolocator.geocode(city)
    print("The latitude of the location is: ", location.latitude)
    print("The longitude of the location is: ", location.longitude)

# Initialize geolocator
geolocator = Nominatim(user_agent="city-geocoder")

# Initialize new columns for latitude and longitude
unique_cities_start['Latitude'] = None
unique_cities_start['Longitude'] = None
unique_cities_finish['Latitude'] = None
unique_cities_finish['Longitude'] = None

# Iterate through the DataFrame and update latitude and longitude for start
for index, row in unique_cities_start.iterrows():
    city_name = row['City']
    location = geolocator.geocode(city_name)
    if location:
        unique_cities_start.at[index, 'Latitude'] = location.latitude
        unique_cities_start.at[index, 'Longitude'] = location.longitude

# Iterate through the DataFrame and update latitude and longitude for finish
for index, row in unique_cities_finish.iterrows():
    city_name = row['City']
    location = geolocator.geocode(city_name)
    if location:
        unique_cities_finish.at[index, 'Latitude'] = location.latitude
        unique_cities_finish.at[index, 'Longitude'] = location.longitude

unique_cities_start.to_csv('unique_cities_start_p.csv', index=False)
unique_cities_finish.to_csv('unique_cities_finish_p.csv', index=False)



