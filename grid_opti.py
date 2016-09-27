# -*- coding: utf-8 -*-

# Course Excercise Modelling and Optimization
# Malte Scharf


#%% Package Import and Options

# Import
import matplotlib.pyplot as plt
import pandas as pd

from pyomo.environ import ConcreteModel, Set, Param, Var, Constraint, Objective, Reals, NonNegativeReals, minimize
from pyomo.opt import SolverFactory

# Global Plotting Options
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 30})


#%% Data Import

    # Reads all sheets as dataframe dictionary
input_data = pd.read_excel('input_data.xls', 
                           sheetname=None, # Imports all sheets
                           header=0, 
                           parse_dates=False)
                   
tp = [1, 24]
timesteps = range (tp[0], tp[1])

locations = ['1', '2']

    # Params (loc independent)
params = input_data['params'] 

c_i_trans = params[(params['id'] == 1)].iloc[0]['value'] # yearly investment cost of transmission line in Euro/(MW*a) 
c_i_wind = params[(params['id'] == 2)].iloc[0]['value'] # yearly investment cost of wind turbin in Euro/(MW*a) 
c_i_store = params[(params['id'] == 3)].iloc[0]['value'] # yearly investment cost of storage unit in Euro/(MWh*a) 

c_o_trans = params[(params['id'] == 4)].iloc[0]['value'] # yearly op cost of transmission line in %
c_o_wind = params[(params['id'] == 5)].iloc[0]['value'] # yearly op cost of wind turbin in %
c_o_store = params[(params['id'] == 6)].iloc[0]['value'] # yearly op cost of storage unit in % 

    # Load (loc dependent)
df_load = input_data['load']

arr_load = [df_load[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['load 1 in MWh'],
            df_load[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['load 1 in MWh']]


    # Wind (loc dependent)
df_wind = input_data['wind']

df_wind.set_index('hour', inplace=True)
wind_1 = df_wind[tp[0]:tp[1]]['power 1 in %']
wind_2 = df_wind[tp[0]:tp[1]]['power 2 in %']

 
    
        #%% Optimization
#m = ConcreteModel()
# 
#   
#    # Sets
#
#m.T = Set(ordered=True, initialize=timesteps) # Set of timesteps
#m.L = Set(ordered=True, initialize=locations) # Set of locations
#  
#    # Variables (Timestep independent)
#
#        # Non-Negative constraint implicitly defined by "NonNegativeReals"
#        # Bounds can not be defined with multiple units
#m.trans = Var(within=NonNegativeReals) # Transmission line installed capacity in MW
#m.wind = Var(m.L, within=NonNegativeReals) # Wind installed capacity in MW
#m.store = Var(m.L, within=NonNegativeReals) # Sorage installed capacity in MWh
#
#    # Variables (Timestep and location dependent)
#m.charge = Var(m.T, m.L, within=Reals) # Charge of Storage in MWh
#m.filling_lev = Var(m.T, m.L, within=NonNegativeReals) # Storage filling level
#
#m.overprod = Var(m.T, m.L, within=Reals) # Overproduction in MWh
#
#
#    # Params
#
#        # Timestep Dependent (KANN MAN PARAMETER AUF LOCATION ABHÄNGIG MACHEN)
#m.load_1 = Param(m.T, initialize=dict(zip(timesteps, load_1))) # Demand per timestep
#m.load_2 = Param(m.T, initialize=dict(zip(timesteps, load_2)))
#
#m.windspeed_1 = Param(m.T, initialize=dict(zip(timesteps, wind_1)))
#m.windspeed_2 = Param(m.T, initialize=dict(zip(timesteps, wind_2)))
#
#        # Unit Dependet
#            # Don't neet to be defined, because they are kept sperarately
#
#   # Constraints
#
#def energy_balance_constr_rule(m, t):
#    return(
#            # 
#        m.wind['1']*m.windspeed_1[t] # BESSER ALS "2" INDIZIEREN!!
#        + m.wind['2']*m.windspeed_2[t] 
#        - sum(m.charge[t, l] for l in m.L)
#        - m.load_1[t] 
#        - m.load_2[t] == 0
#
#        )
#m.energy_balance_constr = Constraint(m.T, rule=energy_balance_constr_rule)
#
#
#def overproduction_1_constr_rule(m, t):
#    return(
#        m.overprod[t, '1'] == m.wind['1']*m.windspeed_1[t] 
#                            - m.charge[t, '1']
#                            - m.load_1[t]
#    )
#m.overproduciton_1_constr = Constraint(m.T, rule=overproduction_1_constr_rule)
#
#
#def overproduction_2_constr_rule(m, t):
#    return(
#        m.overprod[t, '2'] == m.wind['2']*m.windspeed_2[t] 
#                            - m.charge[t, '2']
#                            - m.load_1[t]
#    )
#m.overproduciton_2_constr = Constraint(m.T, rule=overproduction_2_constr_rule)
#
#
#def transmission_1to2_constr_rule(m, t):
#    return(
#        m.overprod[t, '1'] - m.overprod[t, '2'] <= m.trans
#    )
#m.transmission_1to2_constr = Constraint(m.T, rule=transmission_1to2_constr_rule)
#
#
#def transmission_2to1_constr_rule(m, t):
#    return(
#        m.overprod[t, '2'] - m.overprod[t, '1'] <= m.trans
#    )
#m.transmission_2to1_constr = Constraint(m.T, rule=transmission_2to1_constr_rule)
#


# filling level 0 ist 0. ab dann immer differenz und kleiner als Maximal

# Zur not die location als Set wieder rausnehmen
