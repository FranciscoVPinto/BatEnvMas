#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:42 2024

@author: omega
"""

import gymnasium as gym

import ray #ray2.0 implementation
from ray import tune, air
from ray.tune import analysis, ExperimentAnalysis, TuneConfig
from ray.tune.experiment import trial

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig #trainer and config
from ray.rllib.env.env_context import EnvContext
#models
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.pre_checks import env

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
from models2 import ActionMaskModel, CCActionMaskModel
from models_rnn import LSTMActionMaskModel



import random

from trainable import *
from obs_wrapper import *

# from shiftenvRLlib_mas import ShiftEnvMas

from auxfunctions_CC import *

# Custom Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)
ModelCatalog.register_custom_model("lstm_model", LSTMActionMaskModel)

from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env
#profiling
import cProfile
import pstats
from pstats import SortKey
from icecream import ic

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

from experiment_test import ExperimentTest
from testenv import TestEnv
from trainable import Trainable
from utilities import ConfigsParser
from community import Community
from state import StateVars
from environment import FlexEnv
from plots import Plots

from experiment_test import SimpleTests
from testenv import BaselineTest, DummyTester

#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'
storage_path='/home/omega/Downloads/ShareIST'

configs_folder=cwd / 'configs'

#%% exp_name + get configs for experiment
exp_name=YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
configs=ConfigsParser(configs_folder, exp_name)

_, file_apps_conf, file_scene_conf, _ ,file_vars,file_experiment, ppo_config=configs.get_configs()

#%% get configs for testing environment
test_config_file=configs_folder / 'test_config.yaml'
test_name=YAMLParser().load_yaml(test_config_file)['test_name']
test_configs=ConfigsParser(configs_folder, test_name)
# between training and testing the difference is the agents config and the problem config
file_ag_conf,_,_,file_prob_conf,_,_,_=test_configs.get_configs()

#%%Make test env
#dataset
gecad_dataset=datafolder / 'dataset_gecad_clean.csv'
        
#%%  Test environment   
n_tests=YAMLParser().load_yaml(test_config_file)['n_baseline_tests']
for k in range(0,n_tests):    

    test_com=Community(file_ag_conf,
                  file_apps_conf,
                  file_scene_conf,
                  file_prob_conf,
                  gecad_dataset)
    
    
    com_vars=StateVars(file_vars)
    
    

    test_env_config={'community': test_com,
                'com_vars': com_vars,
                'num_agents': test_com.num_agents}
       


    print(k)
    folder_name='baseline_' + test_name
    folder=os.path.join(folder_name, f'iter_{k}')
    
    
    tenvi=FlexEnv(test_env_config)
    dummy_tester=DummyTester(tenvi)
    baselinetest=BaselineTest(tenvi, dummy_tester,folder)
    
    full_state, env_state_conc, episode_metrics, filename = baselinetest.test(results_path=resultsfolder)

