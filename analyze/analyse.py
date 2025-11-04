#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:13:37 2024

@author: omega
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:42 2024

@author: omega
"""



#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


#System
import os
from os import path
from pathlib import Path
import sys
import time
import datetime
from datetime import datetime

#Custom functions
# from plotutils import *



import random


#pyomo
# from auxfunctions_opti import *
from pyomo.environ import *
from pyomo.opt import SolverFactory
import scipy.io as sio
import re 
from itertools import compress
from rich import print
from rich.console import Console
from rich.syntax import Syntax

from testings.experiment_test import ExperimentTest
from testings.testenv import TestEnv
from trains.trainable import Trainable
from utils.utilities import ConfigsParser, FolderUtils
from utils.dataprocessor import YAMLParser
from env.community import Community
from env.state import StateVars
from env.environment import FlexEnv
from analyze.analyzer import Analyzer
#paths

cwd=Path.cwd()
datafolder=cwd.parent  / 'Data'
raylog=cwd.parent  / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd.parent / 'Results'
storage_path='/home/omega/Downloads/ShareIST'

configs_folder=cwd.parent  / 'configs'

#%% 
from analyze.analyzer import Analyzer, AnalyzerMulti
from analyze.plots import Plots
exp_group_defs=YAMLParser().load_yaml(configs_folder / 'results_config.yaml')

#%% Create analyzer objects

experiment_name=exp_group_defs['exps']
days_to_plot=exp_group_defs['days_to_plot']
save_file=exp_group_defs['save_file']

analysis_objs=dict.fromkeys(experiment_name)


stats={}
for exp in experiment_name:
    print(exp)
    
    analysis_objs[exp]={}
    for k in exp_group_defs[exp]['experiments']:
        env_name=exp_group_defs[exp]['experiments'][k]['test_env']
        configs=ConfigsParser(configs_folder, env_name)
        
        file_agents_conf,file_apps_conf,file_scene_conf,file_prob_conf,file_vars,file_experiment,ppo_config=configs.get_configs()
        
        # gecad_dataset=datafolder / 'dataset_gecad_clean.csv'
        file=YAMLParser().load_yaml(file_prob_conf)['dataset_file']
        gecad_dataset=datafolder / file
        
        test_com=Community(file_agents_conf,
                      file_apps_conf,
                      file_scene_conf,
                      file_prob_conf,
                      gecad_dataset)
        
        com_vars=StateVars(file_vars)
        test_env_config={'community': test_com,
                    'com_vars': com_vars,
                    'num_agents': test_com.num_agents}
           
        tenv=FlexEnv(test_env_config)
        
        folder=resultsfolder / exp_group_defs[exp]['experiments'][k]['folder']
        folder_bl=resultsfolder / exp_group_defs[exp]['experiments'][k]['baseline_folder']
        folder_opti=resultsfolder / exp_group_defs[exp]['experiments'][k]['opti_folder']
        analysis_objs[exp][k]=Analyzer(folder, folder_bl, folder_opti,tenv)
        
        # plot some days
        for day in days_to_plot:
            
            analysis_objs[exp][k].plot_one_day(day, 'rl','simple',exp, save=save_file)
            # data_day=analysis_objs[exp][k].get_one_day_data(day, 'rl')
            # analysis_objs[exp][k].plot_one_day(day, 'rl','full',exp, save=False)
            # analysis_objs[exp][k].plot_one_day(day, 'opti','full',exp, save=save_file)
            analysis_objs[exp][k].plot_one_day(day, 'opti','simple',exp, save=save_file)
            
    
    analyse_multi=AnalyzerMulti(analysis_objs[exp],exp)
    analyse_multi.plot_multi_joint(x='x_ratio',y='dif',save=save_file)
    
    #plots for paper
    analyse_multi.plot_multi_joint(x='x_ratio',y='dif_simple',save=save_file)
    analyse_multi.plot_boxplot_year_mean_cost_group(save=save_file)
    
    
    # analyse_multi.plot_multi_joint(x='x_ratio',y='gamma',save=save_file)
    analyse_multi.plot_year_mean_cost_per_model(save=save_file)
    # analyse_multi.plot_year_mean_cost_group(save=False)
    
    # analyse_multi.plot_year_cost(save=False)
    
    data_raw=analyse_multi.get_multi_cost_compare()
    data_year=analyse_multi.get_multi_year_data_eq()
    data_year_diff=analyse_multi.get_multi_year_data_diff()
    data_ag=analyse_multi.get_multi_all_compare()
    stats[exp]=analyse_multi.get_exp_stats()

    #data for paper
    data_paper=data_ag.loc[data_ag.index.get_level_values(2).isin([15,11,31,34,1,2])]
    
    
#%%
obj=analysis_objs[exp][k]
# s=obj.state
# t=2880
# s1=s.loc[t:t+96]
# d=359
# obj.plot_one_day(d,'rl')
# obj.plot_one_day(d,'opti')
env_data=tenv.data
env_data[env_data.index.get_level_values(1)==34]