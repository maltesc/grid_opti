# -*- coding: utf-8 -*-

# Course Excercise Modelling and Optimization
# Malte Scharf


#%% Package Import and Options

# Import
import matplotlib.pyplot as plt
import pandas as pd

from pyomo.environ import ConcreteModel, Set, Param, Var, Constraint, Objective, Reals, NonNegativeReals, minimize
from pyomo.opt import SolverFactory

import numpy as np

# Global Plotting Options
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 30})


#%% Data Import

    # Reads all sheets as dataframe dictionary
input_data = pd.read_excel('input_data.xls', 
                           sheetname=None, # Imports all sheets
                           header=0, 
                           parse_dates=False)
                   
tp = [0, 23] # used for indexing
timesteps = range (tp[0], tp[1])

locations = [0, 1]


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

arr_load = np.array([df_load[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['load 1 in MWh'],
            df_load[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['load 1 in MWh']])


    # Wind (loc dependent)
df_wind = input_data['wind']

arr_wind = np.array([df_wind[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['power 1 in %'],
            df_wind[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['power 2 in %']])

    
        #%% Optimization
m = ConcreteModel()
 
   
    # Sets

m.T = Set(ordered=True, initialize=timesteps) # Set of timesteps
m.L = Set(ordered=True, initialize=locations) # Set of locations
  
    # Variables (Timestep independent)

        # Non-Negative constraint implicitly defined by "NonNegativeReals"
        # Bounds can not be defined with multiple units
m.trans = Var(within=NonNegativeReals) # Transmission line installed capacity in MW
m.wind = Var(m.L, within=NonNegativeReals) # Wind installed capacity in MW
m.store = Var(m.L, within=NonNegativeReals) # Sorage installed capacity in MWh

    # Variables (Timestep and location dependent)
m.charge = Var(m.L, m.T, within=Reals) # Charge of Storage in MWh
m.filling_lev = Var(m.L, m.T, within=NonNegativeReals) # Storage filling level (before timestep)

m.overprod = Var(m.L, m.T, within=Reals) # Overproduction in MWh


   # Constraints

def energy_balance_constr_rule(m, t):
    return(
        sum(m.wind[l]*arr_wind[l, t]
        - m.charge[l, t]
        - arr_load[l, t] for l in m.L) == 0
        )
m.energy_balance_constr = Constraint(m.T, rule=energy_balance_constr_rule)


def overproduction_constr_rule(m, t, l):
    return(
        m.overprod[l, t] == m.wind[l]*arr_wind[l, t]
                            - m.charge[l, t]
                            - arr_load[l, t]
    )
m.overproduciton_1_constr = Constraint(m.T, m.L, rule=overproduction_constr_rule)


def transmission_0to1_constr_rule(m, t):
    return(
        m.overprod[0, t] - m.overprod[1, t] <= m.trans
    )
m.transmission_1to2_constr = Constraint(m.T, rule=transmission_0to1_constr_rule)


def transmission_1to0_constr_rule(m, t):
    return(
        m.overprod[1, t] - m.overprod[0, t] <= m.trans
    )
m.transmission_2to1_constr = Constraint(m.T, rule=transmission_1to0_constr_rule)

def filling_level_constr_rule(m, l, t):
    if t == 0:
        return (m.filling_lev[l, t] == 0)
    else:
        return(
            m.filling_lev[l, t] == m.filling_lev[l, (t-1)] + m.charge[l, (t-1)]
            )
m.filling_level_constr = Constraint(m.L, m.T, rule=filling_level_constr_rule)

def max_filling_level_constr_rule(m, l, t):
    
    return(
        m.filling_lev[l, t] <= m.store[l] # Smaller than installed
            )
m.max_filling_level_constr = Constraint(m.L, m.T, rule=max_filling_level_constr_rule)

    # Objective function
def obj_rule(m):
    return(
        m.trans*(1+c_o_trans)*c_i_trans 
            + sum(  m.wind[l]*(1+c_o_wind)*c_i_wind 
                    + m.store[l]*(1+c_o_store)*c_i_store
                  for l in m.L ) # sum over Locations

            )


m.costs = Objective(sense=minimize, rule=obj_rule)

    # Define Solver
opt = SolverFactory('glpk')

    # Solve the model
results = opt.solve(m, tee=True)

    # Load results back into model
m.solutions.load_from(results)
