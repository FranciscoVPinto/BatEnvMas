#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:25:45 2025

@author: omega
"""

import sys



import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re

from utils.dataprocessor import GecadDataProcessor, YAMLParser
from utils.utilities import utilities

from dp.tests import DPTests


class DPEnergyData:
    """
    Class to inject Differentially Private noise in the gecad datseries.

    Attributes
    ----------
    parser : YAMLParser
        YAML parser instance for configuration loading.
    data : pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    config : dict
        DP parameters.

    Methods
    -------
    get_clean_data(file) -> pd.DataFrame:
        Imports the clean GECAD file as a DataFrame.
    """

    def __init__(self, file, config):
        """
        Initialize the DPEnergyData instance.

        Parameters
        ----------
        file : str
            Path to the GECAD CSV file.
        config : str
            Path to the YAML configuration file.

        """
        self.parser = YAMLParser()
        self.file=file
        self.data = self.get_clean_data()
        self.config = self.parser.load_yaml(config)
        
        self.noisy_data = None
        self.stats=None
        self.stats_daily=None
        
        self.mech_to_use=self.config['mech_to_use']
        self.mechanism_map = {
            "laplace": self.laplace_mechanism,
            "laplace_year": self.laplace_mechanism_year,
            "laplace_day": self.laplace_mechanism_day
        }
        
        self.tests=DPTests

    def get_clean_data(self) -> pd.DataFrame:
        """
        Imports the clean GECAD file and returns it as a DataFrame.

        Parameters
        ----------
        file : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            The imported data as a pandas DataFrame, indexed by the first column.
        """
        return pd.read_csv(self.file, index_col=0)
    
    def get_adjancent_data(self):
        """
        Produces and adjacent datasetr relative to self.data by changing a 
        value each day of the dataset
        """
        
        data_adj=self.data.copy()
        w=96
        
        num_days=int(len(self.data)/w)
        
        cols=[c for c in data_adj.columns if c != 'minutes']
        for d in range(num_days):
            t_init=w*d
            t_end=t_init+w
            for col in cols:
                i=random.randint(t_init, t_end)
                data_adj.loc[i,col]=0
                
            
        return data_adj
            
        
        
        
        
    
    def get_daily_data(self,data):
        """
        Returns a dataframe  with daily values for consumption, peak, +++
        
        Assumes the following sensiitivity computing expressions:
        
        
        """
        data=self.data.copy()
        
        start_date = '2023-01-01 00:00'
        data['datetime'] = pd.to_datetime(start_date) + pd.to_timedelta(data.index * 15, unit='m')
        data.set_index('datetime', inplace=True)
        
        # Daily aggregations:
        daily_stats = data.resample('D').agg(['sum', 'mean', 'min', 'max'])
        
        # import pdb
        # pdb.pdb.set_trace()
        
        daily_stats=daily_stats.copy()
        for col in daily_stats.columns.levels[0]:
            # daily_stats.loc[:, (col, 's_max')] = daily_stats.loc[:, (col, 'max')] - daily_stats.loc[:, (col, 'min')]
            # daily_stats.loc[:, (col, 's_mean')] = daily_stats.loc[:, (col, 'max')] - daily_stats.loc[:, (col, 'mean')]
       
        
            # new_cols = {(col, 's_max'): daily_stats.loc[:, (col, 'max')] - daily_stats.loc[:, (col, 'min')]}
            new_cols = {(col, 's_max'): daily_stats.loc[:, (col, 'max')] - daily_stats.loc[:, (col, 'min')],
                        (col, 's_mean'): (daily_stats.loc[:, (col, 'max')] - daily_stats.loc[:, (col, 'mean')])}


            daily_stats = pd.concat([daily_stats, pd.DataFrame(new_cols)], axis=1)
        
        
        daily_stats = daily_stats.sort_index(axis=1)
        
        #roundings
        daily_stats = daily_stats.mask(daily_stats.abs() < 1e-12, 0.0)
        daily_stats = daily_stats.round(decimals=3)
        
        daily_stats['day'] = daily_stats.index.dayofyear
        daily_stats.set_index('day', inplace=True)
        daily_stats=daily_stats.drop(columns='minutes')
        
        daily_stats = daily_stats.reset_index(drop=True)
        
        return daily_stats
        
    def get_sens_energy_cons(self):
        """
        Computes sensitivity given by:
            
            s=sum(E_t)
            
            
        returns a sensitivity value for every day    
        """
        
        
    
    def laplace_mechanism(self):
        """"
        simple laplace mechanism
        
        - The output must be the noise to add in function self.apply_noise()
        
        - The output of this mechanism must have the same shape as the original data
        
        - Adds noise to PV also
        
        """
        
        laplace_config=self.config.get('laplace')
        
        sensitivity=laplace_config.get('sensitivity')
        epsilon=laplace_config.get('epsilon')
        mean=laplace_config.get('mean')
        
        scale=sensitivity/epsilon
        
        size=self.data.shape
        noise=np.random.laplace(mean,scale,size)
        
        # self.tests.test_structure_equal(noise,self.data)
        
        return noise
        
    def laplace_mechanism_year(self):
        """"
        simple laplace mechanism with sensitivity given by different methods 
        
        - Sensitivity is the same for the wholle year
        
        - The output must be the noise to add in function self.apply_noise()
        
        - The output of this mechanism must have the same shape as the original data
        
        - Adds noise to PV also
        """
        data=self.data
        
        laplace_config=self.config.get('laplace_year')
        epsilon=laplace_config.get('epsilon')
        mean=laplace_config.get('mean')
        s_def=laplace_config.get('sensitivity')
        
        stats=data.describe()
        stats.loc['s_max']=stats.loc['max']-stats.loc['min']
        stats.loc['s_mean']=stats.loc['mean']-stats.loc['max']
        self.stats=stats
        
        sensitivity=stats.loc[s_def]
                
        noise=data.copy()
        for index in sensitivity.index:    
            scale=sensitivity.loc[index]/epsilon
            noise[index]=np.random.laplace(mean,scale,len(noise[index]))
        
        
        self.tests.test_structure_equal(noise,data)
        return noise
    
    
    
    def laplace_mechanism_day(self):
        """"
        simple laplace mechanism with sensitivity given by different methods 
        
        - Sensitivity is different every day
        
        - The output must be the noise to add in function self.apply_noise()
        
        - The output of this mechanism must have the same shape as the original data
        
        - Adds noise to PV also
        """
        data=self.data
        
        laplace_config=self.config.get('laplace_day')
        epsilon=laplace_config.get('epsilon')
        mean=laplace_config.get('mean')
        s_def=laplace_config.get('sensitivity')
        
        stats=self.get_daily_data(data)
        self.stats_daily=stats
        
        w=96
        noise=data.copy()
        
        cols=data.columns
        cols=cols.drop('minutes')

        for day in stats.index:
            t_init=w*day
            t_end=t_init+w
            for col in cols:
                # import pdb
                # pdb.pdb.set_trace()
                sensitivity=stats.loc[day,(col,s_def)]
                scale=sensitivity/epsilon
                # print('col',col)
                # print('sensitivity', sensitivity)
                # print('scale: ', scale)
                # print('day: ', day)
                noise.loc[t_init:t_end-1,col]=np.random.laplace(mean,scale,w)
        
        

        self.tests.test_structure_equal(noise,data)
        return noise
        
    
    def laplace_mechanism_trunc_day(self):
        """"
        Truncated laplace mechanism with sensitivity given by different methods 
        
        - Sensitivity is different every day
        
        - The output must be the noise to add in function self.apply_noise()
        
        - The output of this mechanism must have the same shape as the original data
        
        - Adds noise to PV also
        """
        data=self.data
        
        laplace_config=self.config.get('laplace_day')
        epsilon=laplace_config.get('epsilon')
        mean=laplace_config.get('mean')
        s_def=laplace_config.get('sensitivity')
        
        stats=self.get_daily_data(data)
        self.stats_daily=stats
        
        w=96
        noise=data.copy()
        
        cols=data.columns
        cols=cols.drop('minutes')

        for day in stats.index:
            t_init=w*day
            t_end=t_init+w
            for col in cols:
                sensitivity=stats.loc[day,(col,s_def)]
                scale=sensitivity/epsilon
                noise.loc[t_init:t_end-1,col]=np.random.laplace(mean,scale,w)
        
        

        self.tests.test_structure_equal(noise,data)
        return noise
    
    def apply_dp_noise(self) -> pd.DataFrame:
        """
        Applies differential privacy noise to the dataset based on configuration parameters.
        - PV data is not affected

        Returns
        -------
        pd.DataFrame
            Perturbed dataset after applying differential privacy noise.
            
        """ 

        noise_func=self.mechanism_map.get(self.mech_to_use)
        noise=noise_func()
            
        
        noisy_data = self.data.copy()+noise
        noisy_data['minutes']=self.data['minutes'] # remake again the timeslot column
        noisy_data=noisy_data.clip(lower=0) #There are no negative values in load or generation

        self.noisy_data = noisy_data
        
        pv_cols=[cols for cols in self.data.columns if 'PV' in cols]
        noisy_data[pv_cols]=self.data[pv_cols] # PV not affected
        
        assert noisy_data.shape == self.data.shape, "Noisy data is not the same shape as original data"

        return noisy_data
    
    def plots_compare(self,day_num,var):
        
        
        w=96
        t_init=w*day_num
        t_end=t_init+w
        data=self.data.loc[t_init:t_end-1][var]
        noisy_data=self.apply_dp_noise().loc[t_init:t_end-1][var]

        epsilon=self.config.get(self.mech_to_use)['epsilon']
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='original')
        plt.plot(noisy_data, label='noisy')
        plt.ylabel('kW')
        plt.legend()
        plt.title(f'epsilon={epsilon} | day: {day_num} | dp mech: {self.mech_to_use} ')
        plt.show()
    
    def plots_multi_compare(self,day_num,var, file_list):
        """
        Plots several gecad datasets together 
        (different epsilon variations of the same dataset)
        """
        w=96
        t_init=w*day_num
        t_end=t_init+w
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        for file in file_list:
            file=self.file.parent / file
            data=pd.read_csv(file, index_col=0)
            
            label=utilities().get_num_from_str(file.stem)
            # import pdb
            # pdb.pdb.set_trace()
            plt.plot(data.loc[t_init:t_end-1][var], label=str(label))
        
        # plt.plot(noisy_data, label='noisy')
        plt.ylabel('kW')
        plt.legend()
        # plt.title(f'epsilon={epsilon} | day: {day_num} | dp mech: {self.mech_to_use} ')
        plt.show()
    
    def save_data(self):
        noisy_data=self.apply_dp_noise()
        config=self.config.get(self.mech_to_use)
        
        filename = (
                    self.file.stem + '_'
                    + self.mech_to_use + '_'
                    + 'sens' + '_' + str(config.get('sensitivity')) + '_'
                    + 'eps' + '_' + str(config.get('epsilon'))
                    + '.csv'
                )

        
        name=self.file.parent /  filename
        
        noisy_data.to_csv(name)
        print('saved file:', name)
        pass
        
        
        





