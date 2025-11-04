#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:23:50 2024

@author: omega

--------

    - Holds the classes needed to analysze the test results from a sucessfull experiment
    
    - These results are produced by the test/test.py script

"""

from analyze.plots import Plots
from utils.utilities import FolderUtils, utilities, ConfigsParser
import pandas as pd
from os import path
from pathlib import Path
from utils.dataprocessor import YAMLParser
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


cwd=Path.cwd()
datafolder=cwd.parent / 'Data'
raylog=cwd.parent / 'raylog'
configs_folder=cwd.parent / 'configs'
algos_config = configs_folder / 'algos_configs'
resultsfolder=cwd.parent / 'Results'


class Analyzer():
    """
    From results_folder produces an object with all the needed data and methods to perform analysis 
    
    
    Attributes
    ----------
    folder: path
        An object of pathlib.Path containing the absolute path of the folder with the test results.
    plots_folder: path
        An object of pathlib.Path containing the absolute path of the sub-folder to store plots
    baseline_folder : path
        An object of pathlib.Path containing the absolute path of the folder with the results of the baseline
    opti_folder : path
        An object of pathlib.Path containing the absolute path of the folder with the results of the optimal model 
    env: FlexEnv
        A FlexEnv object
    plot: Plots
        A Plots object
    
    """
    
    def __init__(self,results_folder,
                 baseline_folder,
                 opti_folder,
                 test_env):
        
        
        """        
        Parameters
        ----------
        results_folder: path
            An object of pathlib.Path containing the absolute path of the folder with the test results.
        baseline_folder : path
            An object of pathlib.Path containing the absolute path of the folder with the results of the baseline
        opti_folder : path
            An object of pathlib.Path containing the absolute path of the folder with the results of the optimal model
        test_env: FlexEnv
            A FlexEnv object
        
        """
        
        self.folder=results_folder
        self.plots_folder=self.folder / 'plots'
        FolderUtils().make_folder(self.plots_folder)
        
        self.baseline_folder=baseline_folder
        self.opti_folder=opti_folder
        self.plot=Plots()
        self.get_full_test_data() #import data
        self.env=test_env
        self.get_exp_info()
        

        print(f'Experiment in analysis: {self.folder.name}')
        print(f'Baseline in analysis: {self.baseline_folder.name}')
        print(f'Optimal Solution in analysis: {self.opti_folder.name}')
    
        
    def get_exp_info(self):
        """"
        creates a analyzer attribute with info on the experiment of the results 
        from experiment configs
        
        
        - edit self.exp_info for more info from configs         
        """
    
        self.exp_info=utilities.get_exp_from_results_name(self.folder.name)

        self.config=ConfigsParser(configs_folder, self.exp_info['train_exp'])
        
        _,_,_,_,_,exp_config,algo_config=self.config.get_configs()
        algo_config=YAMLParser().load_yaml(algo_config)
        exp_config=YAMLParser().load_yaml(exp_config)
        
        self.exp_info['model']= exp_config['algorithm']['name']
        
        
    
    def get_full_test_data(self):
        """
        - Scans the results folder and gets metrics and solutions for post analysis and visualization
        
        - Scans the optimization results folder
        
        """
        
        files=FolderUtils().get_file_in_folder(self.folder,'.csv')
        opti_files=FolderUtils().get_file_in_folder(self.opti_folder,'.csv')
        #import testing results data
        for f in files:
            if 'metrics' in f:
                self.metrics=pd.read_csv(f)
                self.metrics['day']=self.metrics['day'].astype(int)
                self.metrics.set_index(['Unnamed: 0','day'], inplace=True)
                # self.metrics=pd.read_csv(f,index_col=['Unnamed: 0'])
                self.metrics.index.names = ['agent', 'day']
                # self.metrics['day'] = self.metrics['day'].astype(float).astype(int)
            elif 'env_state' in f:
                self.state=pd.read_csv(f,index_col='tstep')
            elif 'progress' in f:
                self.progress=pd.read_csv(f)
        # import optimization solutions
        for f in opti_files:
            if 'solutions' in f:
                self.opti_sol=pd.read_csv(f, index_col='tstep')
            elif 'objectives' in f:
                self.opti_objective=pd.read_csv(f,index_col=['tstep', 'day'])
        
        # Import baselines
        self.baseline=self.get_baseline_costs()
    
    def get_one_day_data(self,day_num,source):
        """extract one day from env_state
        - day_num must be the day of the year independentelly of the number of days the dataset has
        
        """
        w=self.env.Tw
        
        if source=='opti':
            data=self.opti_sol
        elif source=='rl':
            data=self.state
        # elif source=='baseline':
        #     data=self.baseline_state
            
        t_init=w*day_num
        assert t_init in self.env.allowed_inits, f"The day ({day_num}) does not exist in the experiment dataset"
        t_end=t_init+w

        return data.loc[t_init:t_end-1]

       
    def get_baseline_costs(self):
 
        # baseline_files=FolderUtils().get_file_in_folder(self.baseline_folder,'.csv')
        folders=FolderUtils().get_subfolders(self.baseline_folder)
        # baseline_files=FolderUtils().get_csv_files_in_subfolders(self.baseline_folder,'.csv')
        
        baseline_costs=[]
        n=0
        
        for fol in folders:
            baseline_files=FolderUtils().get_file_in_folder(fol, 'csv')
            # import pdb
            # pdb.pdb.set_trace()
            for f in baseline_files:
                if 'metrics' in f:
                    self.baseline_metrics=pd.read_csv(f,index_col=0,decimal=',')
                    self.baseline_metrics=self.baseline_metrics.drop(columns='season')
                    self.baseline_metrics=self.baseline_metrics.astype(float)
                    self.baseline_metrics['day'] = self.baseline_metrics['day'].astype(float).astype(int)
                    baseline_costs.append(self.baseline_metrics.loc['com'][['day','cost']])
                    n+=1
                    # df=self.baseline_metrics.join(self.baseline_metrics)
            
        print(f'Baseline mean computed from {n} runs')
                
        day=self.baseline_metrics.loc['com']['day']
        df=pd.concat(baseline_costs, axis=1)
        df_costs=df.drop(columns='day')
        df['cost_mean_base']=df_costs.mean(axis=1)
        df=df['cost_mean_base']
        # df=pd.merge(df,day,left_index=df.index)
        df_final=pd.concat([df,day],axis=1)
        df_final.set_index('day', inplace=True)
        return df_final
        
    def get_cost_compare(self):
        """
        creates a dataframe that stores cost values between optimal solution 
        and rl agent alongside metrics derived from this comparison
        
        if there are day in which the RL solution is better than optimal 
        that means that at least one machine did not turn on on that day 
        during the testing of policies
        
        The result filters out those days and returns only the good days 
        (days in which all appliances)
        
        """
        
        df=pd.merge(self.metrics.loc['com'], 
                    self.opti_objective, 
                    left_index=True, right_index=True)
        
        df=pd.merge(df, self.baseline, left_index=True, right_index=True)
        # import pdb
        # pdb.pdb.set_trace()
        df_compare=df[['cost','objective','cost_mean_base','x_ratio']]
        df_compare=df_compare.copy()
        df_compare=df_compare.astype(float)
        
        a=df_compare['cost'].astype(float)
        b=df_compare['objective']
        c=df_compare['cost_mean_base']
        
        # df_compare.loc[:, 'dif']=((df_compare['cost'].astype(float))-df_compare['objective'])
        df_compare.loc[:, 'dif']=a-b
        df_compare.loc[:, 'dif_simple']=((a+1)-(b+1))/(b+1)
        df_compare.loc[:, 'dif_simple_2']=(((a-b)+1)/(b+1))
        df_compare.loc[:, 'save_rate']=((c-a)/(c-b))*100
        df_compare.loc[:, 'sigma2_0']=abs((b-a)/c)
        df_compare.loc[:, 'sigma2']=(1-df_compare['sigma2_0'])*100
        
        
        utilities().print_info('only works if agenst have the same tariff')
        max_cost=max(self.env.com.agents['ag1'].tariff)*self.env.agents_params['E_prof'].sum()      
        df_compare.loc[:, 'cost_norm']=a/max_cost        
        # utilities().print_info('Random baseline cost is used in indicators')
        
        # df_compare.loc[:, 'gamma'] = np.where(df_compare['objective'] == 0 and df_compare['objective'] != 0, df_compare['objective'])
        # df_compare.loc[:, 'gamma'] = np.where(b == 0 and a == 0, 0)
        # df_compare.loc[:, 'gamma'] = np.where(b > 0 and a > 0, (a-b)/b)
        def compare_values(df):
            # df = df.mask(df_compare.abs() < 1e-7, 0)
            a=df['cost'].astype(float)
            b=df['objective']
            
            if a > 0 and b == 0:
                return a
            elif b == 0 and a == 0:
                return 0
            elif b > 0 and a > 0:
                return (a-b)/b
        
        def gt(df):
            a=40
            if df['gamma']>a:
                return a
            else:
                return df['gamma']
            
        
        
        df_compare = df_compare.mask(df_compare.abs() < 1e-10, 0)
        
        
        df_compare.loc[:, 'gamma']=df_compare.apply(compare_values,axis=1)
        df_compare.loc[:, 'gamma']=df_compare.apply(gt,axis=1)
        # df_compare['gamma'] = df_compare['gamma'].mask(df_compare.abs() > 100, 100)
        
        
        
        df_compare=df_compare.round(3)
        
        if not (df_compare['objective']<=df_compare['cost']).all():
            print(df_compare[df_compare['objective'] > df_compare['cost']])
            print("Some values in 'objective' are greater than 'cost'")
            df_compare = df_compare[df_compare['objective'] <= df_compare['cost']]
            # raise ValueError("Some values in 'objective' are greater than 'cost'")

        
        return df_compare
        
    
    def get_per_agent_costs(self):
        data = []
        indexes=self.metrics.index.unique()
        import pdb
        pdb.pdb.set_trace()
        
        for i in indexes:
            # Extract data for each unique index value
            df = self.metrics.loc[i, ['cost', 'test_epi']].copy()
            # Rename the 'cost' column to the unique index value
            df.rename(columns={'cost': i}, inplace=True)
            # Set 'test_epi' as index
            df.set_index('test_epi', inplace=True)
            data.append(df)
        
        # Concatenate all DataFrames along the columns axis
        df_final = pd.concat(data, axis=1)
        
        #erase the com column
        df_final.drop(columns=['com'], inplace=True)
        df_final=df_final.astype(float)
        df_final = df_final.round(2)
        
        utilities().print_info('hardcoded for the 3 agents and the specific testing community// Need to get community info in analyse phase')
        
        
        utilities().print_info('HardCode alert :Need to solve')
        w=pd.DataFrame([3.6,3.0,3.0],index=df_final.columns,columns=['En'])
        
        new_cols=[]
        for ag in df_final.columns:
            new_col=ag+'_E'
            new_cols.append(new_col)
            df_final[new_col]=df_final[ag]/w.loc[ag]['En']
        
        
        # file_name=str(self.folder / f'cost_compare_{len(indexes)}')
        # self.plot.make_compare_plot2(df_final,filename_save=file_name)
        
        return df_final
    
    
    
    def plot_joint(self,x,y, save=False):
        data=self.get_cost_compare()
        file_name=None
        if save:
            file_name=str(self.folder / f'Joint_Plot_Indicator_{x}_{len(data)}')
        
        self.plot.make_jointplot(data,x,y,filename_save=file_name)
        
    
    
    def plot_one_day(self,day_num,source,plot_type,exp,save=False):
        data=self.get_one_day_data(day_num, source)
        
        # import pdb
        # pdb.pdb.set_trace()
        
        model = 'opti' if source == 'opti' else self.exp_info['model']
        
        
        # model=self.exp_info['model']
        infos={'source':source,
               'day_num':day_num,
               'model':model,
               'experiment':exp}
        
        file_name=None
        if save:
            file_name=str(self.plots_folder / ('profile_' + f'day_{day_num}_'+f'source_{source}'+f'_{model}'+f'_exp_{exp}'))
            

        
        if plot_type=='full':
            self.plot.makeplot_bar_full(data,infos, file_name)
        elif plot_type=='simple':
            self.plot.makeplot_bar_simple(data,infos, file_name)
        


    def plot_cost_hist(self,save=False):
        data=self.get_per_agent_costs()
        
        for k in data.columns:
            if 'E' not in k:
                data=data.drop(columns=k)
                
        file_name=None
        if save:
            file_name=str(self.folder / 'per_agent_cost_histogram.png')
        
        
        self.plot.make_histogram(data, file_name)
        

            
            
            
        
        # data=[]
        # for i in self.metrics.index.unique():
        #     df=self.metrics.loc[i, ['cost', 'test_epi']]
        #     df.rename(columns={'cost': i}, inplace=True)
        #     data.append(df)
    
        # for k in data:
        #     k.set_index('test_epi', inplace=True)
            
        # df_final=pd.concat(data, axis=1)
        # return df_final


 


class AnalyzerMulti():
    "Class to compare experiments"
    def __init__(self,objs,name):
        """gets a list of analyzer experiment objects and performs comparison between experiments"""
        self.results_folder=Path.cwd() / 'Results'
        self.name=name
        # self.config=YAMLParser().load_yaml(results_config)[self.name]
        # self.descript=self.config['description']
        # self.baseline=self.results_folder / self.config['baseline_name']
        self.analyser_objs=objs
        self.plot=Plots()
        self.compare_results_folder=self.results_folder / self.name
        FolderUtils().make_folder(self.compare_results_folder)
        
        if self.name=='double_tars':
            self.label='Double Rate'
        elif self.name=='flat_tars':
            self.label='Simple Rate'
        else:
            self.label='other'
        

    
    def get_multi_cost_compare(self):
        """Aggregates daily data for all experiments in the AnalyzerMulti object
        
        - Returns a dict with a dataframe for every experiment key"""
        data={}
        opti_source=[]
        base_source=[]
        for label in self.analyser_objs:
            print(label)
            data[label]=self.analyser_objs[label].get_cost_compare()
            opti_source.append(self.analyser_objs[label].opti_folder.name)
            base_source.append(self.analyser_objs[label].baseline_folder.name)
            
        if len(set(opti_source))==1:
            self.opti_unique=True
        else: 
            self.opti_unique=False
        if len(set(base_source))==1:
            self.base_unique=True
        else:
            self.base_unique=False

        
        return data
      
        
    def get_multi_year_data_eq(self):
        """
        To when optimal solution is the same for all experiments under the 
        the same experiment group
        
        returns data for the wholle year for every experiment in the AnalyzerMulti 
        object when optimal values and base\line values are differente for each experiment 
        
        """
        # data=self.get_multi_cost_compare()
        data=self.get_multi_all_compare()
        exp_indexes=data.index.get_level_values('experiment').unique()
        df=pd.DataFrame(index=exp_indexes,columns=['cost_mean','cost_total'])
        
        for key in exp_indexes:
            df.loc[key,'cost_mean']=data.loc[key]['cost'].mean()
            df.loc[key,'cost_total']=data.loc[key]['cost'].sum()
        
        return df
            
        
    def get_multi_year_data_diff(self):
        """
        To use when optimal solution is differente between experiments under
        the same experiment group
        
        returns data for the wholle year for every experiment in the AnalyzerMulti 
        object when optimal values and baseline values are differente for each experiment 
        
        """
        # data=self.get_multi_cost_compare()
        data=self.get_multi_all_compare()
        exp_indexes=data.index.get_level_values('experiment').unique()
        df=pd.DataFrame(index=exp_indexes,columns=['cost_mean',
                                                    'objective_mean',
                                                    'cost_mean_base_mean',
                                                    'cost_total',
                                                    'objective_total',
                                                    'cost_mean_base_total',
                                                    'opti_dif'])
        
        
        for key in exp_indexes:
            df.loc[key,'cost_mean']=data.loc[key]['cost'].mean()
            df.loc[key,'objective_mean']=data.loc[key]['objective'].mean()
            df.loc[key,'cost_mean_base_mean']=data.loc[key]['cost_mean_base'].mean()
            df.loc[key,'cost_total']=data.loc[key]['cost'].sum()
            df.loc[key,'objective_total']=data.loc[key]['objective'].sum()
            df.loc[key,'cost_mean_base_total']=data.loc[key]['cost_mean_base'].sum()
            df.loc[key,'opti_dif']=(df.loc[key]['cost_mean']-df.loc[key]['objective_mean'])/df.loc[key]['objective_mean']
        
        return df
    
    
    def get_multi_all_compare(self):
        """
        - returns a dataframe (derived from get_multi_cost_compare()) with 
        indexes experiment, day, tstep_init
        
        - If experiments were sucessfull in a differente number of days 
        (outputs from get_multi_cost_compare() with different number of rows) 
        it will show only the common days
        
        - If opti values and base values are the same ir creates new indexes 
        opti and base and treats them as experiments
        
        """
        
        dfs = []
        data_raw=self.get_multi_cost_compare()

        data=data_raw.copy()
        label=list(data.keys())[0] #just the random first label
        if self.opti_unique:
            data['opti']=data[label].copy()
            data['opti']['cost']=data['opti']['objective']
        
        if self.base_unique:
            data['base']=data[label].copy()
            data['base']['cost']=data['base']['cost_mean_base']
        

        # Iterate through the dictionary
        for key, df in data.items():
            # Add the dictionary key as a new level in the index
            df = df.assign(experiment=key)
            df = df.set_index('experiment', append=True)
            
            # Reorder the index levels to have 'source' as the first level
            df = df.reorder_levels(['experiment', 'tstep', 'day'])
            
            # Append the processed DataFrame to the list
            dfs.append(df)
        
        # Concatenate all processed DataFrames
        # Step 2: Find common 'day' values across all DataFrames
        common_days = set(dfs[0].index.get_level_values('day'))  # Initialize with the first DataFrame's 'day' values
        for df in dfs:
            common_days.intersection_update(set(df.index.get_level_values('day')))  # Update to keep only common 'day' values
        
        #Filter each DataFrame to include only rows with common 'day' values
        dfs_filtered = [df[df.index.get_level_values('day').isin(common_days)] for df in dfs]
        
        #Concatenate the filtered DataFrames
        result = pd.concat(dfs_filtered)
        
        return result

    

    def get_exp_stats(self):
        data=self.get_multi_all_compare()
        
        data=data[data['x_ratio']<=4] #critical days
        
        stats=data.groupby('experiment').describe()
        
        stats=stats[['cost', 'dif_simple']]
        
        file_name='stats' + self.name + '.csv'
        
        stats.to_csv(self.compare_results_folder / file_name )
        
        return stats
        

    def check_equals(self):
        """
        Check the several dataframes in the experiment and checks if they have the same optimal solution and 
        baseline
        """
        data=self.get_multi_cost_compare()
        
        
        data=self.get_multi_cost_compare()
        columns=['objective','cost_mean_base']
        for key, df in data.items():
            dfs = []
            for col in columns:
                dfs.append(df[col])
                final_data=pd.concat(dfs,axis=1)
            print(final_data)
        
        

    
    def plot_multi_joint(self,x,y, save=False):        
        data=self.get_multi_cost_compare()
        
        
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / f'Joint_Plot_{y}_{self.name}')
        
        infos={'experiment':self.name,
              'title': f"RL vs optimal solution ({self.label})",
              'x_label': 'Daily excess PV availability normalized by appliance energy need (-)',
              'y_label':'€/day',
              'hue':'experiment'}
        
        
        self.plot.make_multi_jointplot(data,x,y,infos,filename_save=file_name)
        
    
    def plot_per_agent_cost_multi_hist(self,save=False):
        data={}
        for obj_id in self.analyser_objs:
            
            data_agent=self.analyser_objs[obj_id].get_per_agent_costs()
            
            for k in data_agent.columns:
                if 'E' not in k:
                    data_agent=data_agent.drop(columns=k)
                    
            data[obj_id]=data_agent

        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'per_agent_cost_histogram.png')
        
        
        self.plot.make_multi_histogram(data, file_name)
        
      
    def plot_year_mean_cost_per_model(self,save=False):
        df=self.get_multi_year_data_eq()
        variable='cost_mean'
        df=df[variable]
        
        
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'daily_mean_costs.png')
   
        infos={}
        infos['title']='Daily Mean Cost for running collective appliances'
        infos['x_label']='model'
        infos['y_label']='€/day' 
        infos['hue']='experiment'
        infos['var']=variable
        self.plot.plot_barplot(df,file_name,infos)
        
    def plot_year_mean_cost_per_model_opti(self,save=False):
        """
        - Plots the mean daily optimal objective for each experiment under the
        same experiment group.
        """
        df=self.get_multi_year_data_diff()
        variable='objective_mean'
        df=df[variable]
        
        
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'daily_mean_costs.png')
   
        infos={}
        infos['title']='Daily Mean Optimal Cost for running collective appliances'
        infos['x_label']='model'
        infos['y_label']='€/day' 
        infos['hue']='experiment'
        infos['var']=variable
        self.plot.plot_barplot(df,file_name,infos)
    
    def plot_year_mean_cost_group(self,save=False):
        """
        Barplot that compares models vs optimal vs baseline in group
        The Optimal values must be the same
        """
        df=self.get_multi_year_data_eq()
        # import pdb
        # pdb.pdb.set_trace()
        # df=df[['cost_mean','objective_mean','cost_mean_base_mean']]
        
        
        # keys=['objective_mean','cost_mean_base_mean']
        # for k in keys:
        #     assert np.all(df[k] == df[k].iloc[0])
        
        # for k in keys:
        #     df.loc[k]=df[k].iloc[0]
        #     df=df.drop(k,axis=1)
            
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'daily_mean_costs.png')
        
        
        infos={}
        infos['title']=f'Daily Mean Cost for all year ({self.name})'
        infos['x_label']='model'
        infos['y_label']='€/day' 
        infos['hue']='experiment'
        # infos['legend']=False
   
        self.plot.plot_barplot(df,file_name,infos)
        
    def plot_boxplot_year_mean_cost_group(self,save=False):
        """
        Boxplot that compares models vs optimal vs baseline in group
        The Optimal values must be the same
        """
        
        
        df=self.get_multi_all_compare()
        
        # df=df[df['x_ratio']<=4]
        # label2='_hard days'
        
        df=df['cost']
        num_days=len(df.index.get_level_values('day').unique())
        
        infos={}
        infos['title']='Distribution of daily costs'
        infos['x_label']='model'
        infos['y_label']='€/day' 
        infos['hue']='experiment'
        
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / f'box_plot_{self.name}_days_{num_days}_exp_{self.name}.png')
        
        
        # infos['legend']=False
   
        self.plot.plot_boxplot(df,file_name,infos)
    
      
    def plot_year_cost(self,save=False):
        df=self.get_multi_year_data()
        df=df[['cost_total','objective_total','cost_mean_base_total']]
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'year_total_costs.png')
        
        infos={}
        infos['title']='Year Total Cost'
        infos['x_label']='model'
        infos['y_label']='€/year'
        infos['hue']='experiment'
        
        self.plot.plot_barplot(df,file_name,infos)
  
    
    def plot_per_agent_year_cost(self,save=False):
        data={}
        for obj_id in self.analyser_objs:
            data_agent=self.analyser_objs[obj_id].get_per_agent_costs()
            
            for k in data_agent.columns:
                if 'E' not in k:
                    data_agent=data_agent.drop(columns=k)
            
            data_agent=data_agent.mean()
            # data_agent=data_agent.sum()
            data[obj_id]=data_agent
        
        df=pd.DataFrame(data)
        
        reshaped_df = df.stack().reset_index()
        reshaped_df.columns = ['Label', 'Category', 'Value']
        
            
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'yearlly_mean_per_agent.png')
        
        # return data
        self.plot.plot_multi_bar(reshaped_df,file_name)
            
            
    def plot_multi_metrics(self,save=False):
        data={}
        metrics=['episode_reward_mean']
        for obj_id in self.analyser_objs:
            if obj_id == 'Random_baseline':
                continue  # Skip the specific key
            for m in metrics:
                # import pdb
                # pdb.pdb.set_trace()
                data_agent=pd.DataFrame(self.analyser_objs[obj_id].progress[m]).rolling(300).mean()
                # data_agent[m+'smooth']=data_agent[m].rolling(20).mean()
                data[obj_id]=data_agent
         
        filename_save=None        
        if save:
            filename_save=str(self.compare_results_folder / 'mean_reward.png')
             
             
        self.plot.plotline_smooth(data, filename_save)

  
# class AnalyzerMulti():
#     "Class to compare experiments"
#     def __init__(self,results_config,name):
#         """gets a list of analyzer experiment objects and performs comparison between experiments"""
#         self.results_folder=Path.cwd() / 'Results'
#         self.name=name
#         self.config=YAMLParser().load_yaml(results_config)[self.name]
#         self.descript=self.config['description']
#         self.baseline=self.results_folder / self.config['baseline_name']
#         self.analyser_objs=self.get_analyser_objects()
#         self.plot=Plots()
#         self.compare_results_folder=self.results_folder / self.name
#         FolderUtils().make_folder(self.compare_results_folder)
        
#     def get_experiments_folders(self):
#         exp_folders={}
#         exps=self.config['experiments']
#         for k in exps:
#             exp_folders[exps[k]['label']]=exps[k]['folder']
#         return exp_folders
    
#     def get_analyser_objects(self):
#         exp_dict=self.get_experiments_folders()
#         multi_analyser={}
#         # import pdb
#         # pdb.pdb.set_trace()
#         for label in exp_dict:
#             exp_dict[label]=Analyzer(self.results_folder / exp_dict[label],self.baseline)
        
#         # exp_dict['Random_baseline']=Analyzer(self.baseline, self.baseline)
#         return exp_dict
    
#     def get_multi_cost_compare(self):
#         data={}
#         for label in self.analyser_objs:
#             print(label)
#             data[label]=self.analyser_objs[label].get_cost_compare()
            
#         return data
        
    
    
#     def plot_multi_joint(self,x,y, save=False):        
#         data=self.get_multi_cost_compare()
#         utilities().print_info('removing the random baseline from plot_multi_joint')
#         # data.pop('Random_baseline')

#         file_name=None
#         if save:
#             file_name=str(self.compare_results_folder / f'Joint_Plot_Indicator_{x}_{len(data)}_{self.name}')
        
#         self.plot.make_multi_jointplot(data,x,y,filename_save=file_name)
        
    
#     def plot_per_agent_cost_multi_hist(self,save=False):
#         data={}
#         for obj_id in self.analyser_objs:
            
#             data_agent=self.analyser_objs[obj_id].get_per_agent_costs()
            
#             for k in data_agent.columns:
#                 if 'E' not in k:
#                     data_agent=data_agent.drop(columns=k)
                    
#             data[obj_id]=data_agent

#         file_name=None
#         if save:
#             file_name=str(self.compare_results_folder / 'per_agent_cost_histogram.png')
        
        
#         self.plot.make_multi_histogram(data, file_name)
        
      
#     def plot_year_mean_cost(self,save=False):
#         data=self.get_multi_cost_compare()

#         utilities.print_info('The optimization solutions and objectives must be inside the random folder results')
#         df=pd.DataFrame()
#         for obj_id in data:
#             df[obj_id]=data[obj_id]['cost']
#         df['opti']=data[obj_id]['objective']
#         df['rand']=self.analyser_objs[obj_id].get_baseline_costs()['cost_mean_base'].values
        
#         utilities().print_info('possible bug with random baseline costs vs rand')
#         # df=df.drop(columns=['Random_baseline'])
#         data_sorted=df.mean().sort_values()
#         print(data_sorted)
        
#         file_name=None
#         if save:
#             file_name=str(self.compare_results_folder / 'models_barplot.png')
            
#         # import pdb
#         # pdb.pdb.set_trace()    
#         self.plot.plot_barplot(data_sorted,file_name)
      
#     def plot_year_cost(self,save=False):
#         data=self.get_multi_cost_compare()
    
#         utilities.print_info('The optimization solutions and objectives must be inside the random folder results')
#         df=pd.DataFrame()
#         for obj_id in data:
#             df[obj_id]=data[obj_id]['cost']
#         df['Opti']=data[obj_id]['objective']
#         df['Random']=self.analyser_objs[obj_id].get_baseline_costs()['cost_mean_base'].values
        
#         utilities().print_info('possible bug with random baseline costs vs rand')
#         # df=df.drop(columns=['Random_baseline'])
#         data_sorted=df.sum().sort_values()
#         data_to_plot=pd.DataFrame(data_sorted).transpose()
#         data_to_plot.index=['year_cost']
        
#         print(data_to_plot)
        
#         file_name=None
#         if save:
#             file_name=str(self.compare_results_folder / f'models_barplot_{self.name}.png')
            
#         # import pdb
#         # pdb.pdb.set_trace()    
#         self.plot.plot_barplot(data_to_plot,file_name)
  
    
#     def plot_per_agent_year_cost(self,save=False):
#         data={}
#         for obj_id in self.analyser_objs:
#             data_agent=self.analyser_objs[obj_id].get_per_agent_costs()
            
#             for k in data_agent.columns:
#                 if 'E' not in k:
#                     data_agent=data_agent.drop(columns=k)
            
#             data_agent=data_agent.mean()
#             # data_agent=data_agent.sum()
#             data[obj_id]=data_agent
        
#         df=pd.DataFrame(data)
        
#         reshaped_df = df.stack().reset_index()
#         reshaped_df.columns = ['Label', 'Category', 'Value']
        
            
#         file_name=None
#         if save:
#             file_name=str(self.compare_results_folder / 'yearlly_mean_per_agent.png')
        
#         # return data
#         self.plot.plot_multi_bar(reshaped_df,file_name)
            
            
#     def plot_multi_metrics(self,save=False):
#         data={}
#         metrics=['episode_reward_mean']
#         for obj_id in self.analyser_objs:
#             if obj_id == 'Random_baseline':
#                 continue  # Skip the specific key
#             for m in metrics:
#                 # import pdb
#                 # pdb.pdb.set_trace()
#                 data_agent=pd.DataFrame(self.analyser_objs[obj_id].progress[m]).rolling(300).mean()
#                 # data_agent[m+'smooth']=data_agent[m].rolling(20).mean()
#                 data[obj_id]=data_agent
         
#         filename_save=None        
#         if save:
#             filename_save=str(self.compare_results_folder / 'mean_reward.png')
             
             
#         self.plot.plotline_smooth(data, filename_save)
        
        
        
            
        
        
                
        
        
        
        
