#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 09:35:26 2025

@author: omega

This script generates Differentially private datasets using the DPEnergyData class

"""
from pathlib import Path
from dp.diffpriv import DPEnergyData

cwd=Path.cwd()
datafolder=cwd.parent / 'Data'
configs_folder=cwd.parent / 'configs'
algos_config = configs_folder / 'algos_configs'
resultsfolder=cwd.parent / 'Results'

config=configs_folder / 'dp' / 'dp_config.yaml'
file=datafolder / 'dataset_gecad_clean.csv'


#%% 
dp=DPEnergyData(file, config)
dp.save_data()
# data_Adj=dp.get_adjancent_data()
data=dp.data
noisy_data=dp.apply_dp_noise()

# noise=dp.laplace_mechanism()
# noise=dp.laplace_mechanism_year()

# noise=dp.laplace_mechanism_day()

daily_stats=dp.get_daily_data(dp.data)
dp.plots_compare(1, 'PV3')



# day_d_a=dp.get_daily_data(dp.get_adjancent_data())

# ag1=dp.data.loc[0:96]['ag1']
# ag1_a=data_Adj.loc[0:96]['ag1']

# np.random.laplace(0.0,0.0)


