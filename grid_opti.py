# -*- coding: utf-8 -*-

# Course Excercise Modelling and Optimization
# Malte Scharf


#%% Package Import and Options

# Import
import matplotlib.pyplot as plt
import pandas as pd

# Global Plotting Options
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 30})


#%% Data Import

    # Reads all sheets as dataframe dictionary
input_data = pd.read_excel('input_data.xls', 
                           sheetname=None, # Imports all sheets
                           header=0, 
                           parse_dates=False)
                   
tp = [1, 8760] # used for indexing
timesteps = range (tp[0], tp[1])

    # Params (loc independent)
params = input_data['params'] 

c_i_t = params[(params['id'] == 1)].iloc[0]['value'] # yearly investment cost of transmission line in Euro/(MW*a) 
c_i_w = params[(params['id'] == 2)].iloc[0]['value'] # yearly investment cost of wind turbin in Euro/(MW*a) 
c_i_s = params[(params['id'] == 3)].iloc[0]['value'] # yearly investment cost of storage unit in Euro/(MWh*a) 

c_o_t = params[(params['id'] == 4)].iloc[0]['value'] # yearly op cost of transmission line in %
c_o_w = params[(params['id'] == 5)].iloc[0]['value'] # yearly op cost of wind turbin in %
c_o_s = params[(params['id'] == 6)].iloc[0]['value'] # yearly op cost of storage unit in % 

    # Load (loc dependent)
load = input_data['load']

load.set_index('hour', inplace=True)
load_1 = load[tp[0]:tp[1]]['load 1 in MWh']
load_2 = load[tp[0]:tp[1]]['load 2 in MWh']

    # Wind (loc dependent)
wind = input_data['wind']

wind.set_index('hour', inplace=True)
wind_1 = wind[tp[0]:tp[1]]['power 1 in %']
wind_2 = wind[tp[0]:tp[1]]['power 2 in %']

print(wind_1)
