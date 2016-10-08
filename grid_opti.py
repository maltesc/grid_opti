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
import os

# Global Plotting Options
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 30})

# Creates output folders
charts_dir = os.getcwd() + "/output_charts"
if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
tables_dir = os.getcwd() + "/output_tables"
if not os.path.exists(tables_dir):
            os.makedirs(tables_dir)
            
#%% Data Import

    # Reads all sheets as dataframe dictionary
input_data = pd.read_excel('input_data.xls', 
                           sheetname=None, # Imports all sheets
                           header=0, 
                           parse_dates=False)
                   
tp = [0, 168] # used for indexing
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
                    (df_load['hour'] <= tp[1])]['load 0 in MWh'],
            df_load[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['load 1 in MWh']])


    # Wind (loc dependent)
df_wind = input_data['wind']

arr_wind = np.array([df_wind[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['power 0 in %'],
            df_wind[   (df_load['hour'] >= tp[0]) & 
                    (df_load['hour'] <= tp[1])]['power 1 in %']])

    
        #%% Optimization
m = ConcreteModel()
 
   
    # Sets

m.T = Set(ordered=True, initialize=timesteps) # Set of timesteps
m.L = Set(ordered=True, initialize=locations) # Set of locations
  
    # Variables (Timestep independent)

        # Non-Negative constraint implicitly defined by "NonNegativeReals"
        # Bounds can not be defined with multiple units
m.inst_trans = Var(within=NonNegativeReals) # Transmission line installed capacity in MW
m.inst_wind = Var(m.L, within=NonNegativeReals) # Wind installed capacity in MW
m.inst_store = Var(m.L, within=NonNegativeReals) # Sorage installed capacity in MWh

    # Variables (Timestep and location dependent)
m.charge = Var(m.L, m.T, within=Reals) # Charge of Storage in MWh
m.filling_lev = Var(m.L, m.T, within=NonNegativeReals) # Storage filling level (before timestep)

m.loss = Var(m.L, m.T, within=NonNegativeReals) # Not used overproduction
m.overprod = Var(m.L, m.T, within=Reals) # Overproduction in MWh


   # Constraints

# Energy must be balanced (no losses)
def energy_balance_constr_rule(m, t):
    return(
        sum(m.inst_wind[l]*arr_wind[l, t]/100
        - m.charge[l, t]
        - arr_load[l, t]
        - m.loss[l, t] for l in m.L) == 0
        )
m.energy_balance_constr = Constraint(m.T, rule=energy_balance_constr_rule)

# Overproduction on one side (can be negative)
def overproduction_constr_rule(m, t, l):
    return(
        m.overprod[l, t] == m.inst_wind[l]*arr_wind[l, t]/100
                            - m.charge[l, t]
                            - arr_load[l, t]
                            - m.loss[l, t]
    )
m.overproduciton_1_constr = Constraint(m.T, m.L, rule=overproduction_constr_rule)


# Transmission musst be always lower than maximum transmission capacity (installed)
def transmission_0to1_constr_rule(m, t):
    return(
        m.overprod[0, t] <= m.inst_trans
    )
m.transmission_1to2_constr = Constraint(m.T, rule=transmission_0to1_constr_rule)


def transmission_1to0_constr_rule(m, t):
    return(
        m.overprod[1, t] <= m.inst_trans
    )
m.transmission_2to1_constr = Constraint(m.T, rule=transmission_1to0_constr_rule)


# Filling level is 0 at the beginning. Then dependent on charge (on each side)
def filling_level_constr_rule(m, l, t):
    if t == tp[0]:
        return (m.filling_lev[l, t] == 0)

    else:
        return(
            m.filling_lev[l, t] == m.filling_lev[l, (t-1)] + m.charge[l, (t-1)]
            )
m.filling_level_constr = Constraint(m.L, m.T, rule=filling_level_constr_rule)

def max_filling_level_constr_rule(m, l, t):
    
    return(
        m.filling_lev[l, t] <= m.inst_store[l] # Smaller than installed
            )
m.max_filling_level_constr = Constraint(m.L, m.T, rule=max_filling_level_constr_rule)

    # Objective function
def obj_rule(m):
    return(
        m.inst_trans*(1+c_o_trans)*c_i_trans 
            + sum(  m.inst_wind[l]*(1+c_o_wind)*c_i_wind 
                    + m.inst_store[l]*(1+c_o_store)*c_i_store
                  for l in m.L ) # sum over Locations

            )


m.costs = Objective(sense=minimize, rule=obj_rule)

    # Define Solver
opt = SolverFactory('glpk')

    # Solve the model
results = opt.solve(m, tee=True)


#%% Postprocessig 

print('\nInstalled transmission in MW: ' + str(m.inst_trans.value))
            
df_results = pd.DataFrame()
for loc in locations:

    print('\nInstalled wind ' 
            + str(loc) 
            + ' in MW: ' 
            + str(m.inst_wind[loc].value))
    print('\nInstalled storage ' 
            + str(loc) 
            + ' in MWh: ' 
            + str(m.inst_store[loc].value))

            
    df_results['load '
                + str(loc) 
                + ' in MWh'] = arr_load[loc, tp[0]:tp[1]]
                
    df_results['wind '
                + str(loc) 
                + ' in MWh'] = m.inst_wind[loc].value*arr_wind[loc, tp[0]:tp[1]]/100

    df_results['wind '
                + str(loc) 
                + ' in %'] = arr_wind[loc, tp[0]:tp[1]]
                
    df_results['charge ' 
                + str(loc) 
                + ' in MWh'] = [m.charge[i].value 
                                    for i in m.charge if i[0]==loc]
    df_results['overprod ' 
                + str(loc) 
                + ' in MWh'] = [m.overprod[i].value 
                                    for i in m.overprod if i[0]==loc]
    df_results['loss ' 
                + str(loc) 
                + ' in MWh'] = [m.loss[i].value 
                                    for i in m.loss if i[0]==loc]                                                                     
    df_results['filling_lev ' 
                + str(loc) 
                + ' in MWh'] = [m.filling_lev[i].value 
                                    for i in m.filling_lev if i[0]==loc]

    df_results['transmission_0to1 in MWh'] = [(m.overprod[0, t].value)
                                                for t in timesteps]    
                                                
print (df_results)

#%% Plots

for loc in locations:
    
    plt.figure(figsize=(40, 20))
    
    y0 = df_results['wind '+str(loc)+' in MWh'] - df_results['load '+str(loc)+' in MWh']
    y0_lab = 'Residual load '+str(loc)+' in MWh/h' 
    
    y1 = df_results['charge '+str(loc)+' in MWh/h']
    y1_lab = 'Charge '+str(loc)+' in MWh'
    
    y2 = df_results['loss '+str(loc)+' in MWh/h']
    y2_lab = 'Loss '+str(loc)+' in MWh'  
    
    x0 = y0.index
    
    line = y0.plot(kind='line',
                   #drawstyle='steps',
                   legend=True,
                   color = 'grey',
                   linewidth=4)    
                   
    line.fill_between(x0, y0, where=y0<=0, 
                      alpha=0.5, 
                      interpolate=True, 
                      color='red')
    line.fill_between(x0, y0, where=y0>0, 
                      alpha=0.5, 
                      interpolate=True, 
                      color='green')
     
    y1.plot(kind='line',
            legend=True,
            color = 'yellow',
            linewidth=6)
    y2.plot(kind='line',
            legend=True,
            color = 'red',
            linewidth=6)

                   
    line.legend(labels=[y0_lab, y1_lab, y2_lab], 
                       loc='best')
    
    #line.set_xlabel('Hour of the year')
    line.set_ylabel('Power in MWh/h')
    
    plot_name = 'res_load '+ str(loc)
    plt.savefig(    charts_dir 
                    + '/'
                    + plot_name 
                    + '.png')
    
    plt.clf()
    plt.cla() 
    
    # Filling Level Figure
    plt.figure(figsize=(40, 10))
    
    y4 = df_results['filling_lev '+str(loc)+' in MWh']
    y4_lab = 'Filling level '+str(loc)+' in MWh'    
    
    x0 = y4.index
    
    line = y4.plot(kind='line',
                   #drawstyle='steps',
                   legend=True,
                   color = 'grey',
                   linewidth=4)    
                   
    line.fill_between(x0, y4, where=y4>=0, 
                      alpha=0.5, 
                      interpolate=True, 
                      color='yellow')


                   
    line.legend(labels=[y4_lab], 
                       loc='best')
    
    #line.set_xlabel('Hour of the year')
    line.set_ylabel('Filling level in MWh')
    
    plot_name = 'filling_lev '+ str(loc)
    plt.savefig(    charts_dir 
                    + '/'
                    + plot_name 
                    + '.png')
    
    plt.clf()
    plt.cla() 
    
    # Transmission Figure

plt.figure(figsize=(40, 10))

y3 = df_results['transmission_0to1 in MWh']
y3_lab = 'Transmission 0to1 in MWh/h'

line = y3.plot(kind='line',
        legend=True,
        color = 'blue',
        linewidth=8)    
x0 = y3.index       
line.fill_between(x0, y3, where=y3>=0, 
                  alpha=0.5, 
                  interpolate=True, 
                  color='grey')
line.fill_between(x0, y3, where=y3<0, 
                  alpha=0.5, 
                  interpolate=True, 
                  color='darkgrey')

line.legend(labels=[y3_lab], 
                   loc='best')   
                   
line.set_ylabel('Transmission in MWh/h')

plot_name = 'transmission'
plt.savefig(    charts_dir 
                + '/'
                + plot_name 
                + '.png')

plt.clf()
plt.cla() 