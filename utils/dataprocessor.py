#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:31:38 2024

@author: omega
"""

import pandas as pd
import numpy as np
import yaml
from utils.utilities import utilities
# from metrics import Metrics


class YAMLParser:
    """
    parsing YAML files 
    """        
    def load_yaml(self,file):
        "outputs a dict from the config files"
        with open(file, 'r') as f:
            data = yaml.load(f,Loader=yaml.FullLoader) 
        return data
    
    def write_yaml(self,file, key, value):
        with open(file, 'r') as f:
            data = yaml.load(f,Loader=yaml.FullLoader)
        
        data[key] = value
         
        with open(file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        


class DataProcessor():
    """
    general class for dataprocessing
    """
        
    @staticmethod
    def hours_to_minutes(time_str, resolution_minutes):
        """
        Convert time in hours to timeslot index in a discretized time series.

        Parameters:
            time_str (str): Time in hours. Format can be 'h', 'h:mm', or 'hh:mm'.
            resolution_minutes (int): Resolution of each timeslot in minutes.

        Returns:
            int: Timeslot index.
        """
        # Convert time string to hours and minutes
        if ':' in time_str:
            hours, minutes = map(int, time_str.split(':'))
        else:
            hours = int(time_str)
            minutes = 0
            
        # Calculate total minutes
        total_minutes = hours * 60 + minutes
        
        # Calculate timeslot index
        timeslot_index = total_minutes // resolution_minutes
        
        return total_minutes, timeslot_index
    
    
    
    @staticmethod        
    def make_env_data(data,timesteps, load_id, pv_factor):
        
        """
        Constructs a DataFrame data to feed the environment data provided the load id (reference to the original dataset)
    
        This method extracts a subset of the input time series data and computes the load, PV generation,
        net delta, and excess generation for each timestep.
    
        Parameters
        ----------
        data : pandas.DataFrame
            The input time series data. Must contain columns 'minutes', 'PV', and 'load'.
        timesteps : int
            The number of timesteps to include from the start of the data.
        load_id : str
            The column name in `data` corresponding to the desired load in the original dataset.
        pv_factor : float
            Scaling factor to apply to the PV generation data.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns:
                - 'minutes': Time in minutes for each timestep.
                - 'load': Load data for the specified load_id.
                - 'gen': Scaled absolute PV generation.
                - 'delta': Net load minus generation.
                - 'excess': Excess generation (when generation exceeds load, else zero).
    
        """
        
        df=pd.DataFrame()
        
        df['minutes']=data.iloc[0:timesteps]['minutes']
        df['load']=data.iloc[0:timesteps][load_id]
        df['gen']=pv_factor*abs(data.iloc[0:timesteps]['PV'])
        df['delta']=df['load']-df['gen']
        df['excess']=[max(0,-df['delta'][k]) for k in range(timesteps)] 
        
        
        return df
    
    
    def max_0(self,x):
        return max(0, -x)
    
    
    def make_com_data(self,agent_dict,
                      t_init,
                      t_end,
                      pv_factor):
        """
        This function takes agents objects and outputs a dataframe to be used in the MARL environment
        In this version, each agent is seeying community quantities but also individual quantities
        
        """
    
        total_PV = pd.DataFrame({'gen_com':pv_factor*sum(agent.data['gen'] for agent in agent_dict.values())})
        total_load = pd.DataFrame({'load_com':sum(agent.data['load'] for agent in agent_dict.values())})
        df=pd.concat([total_PV,total_load], axis=1)
        
        utilities().print_info('delta and excess are computed with community agregated load')
        df['delta']=df['load_com']-df['gen_com']
        df['excess']=df['delta'].apply(self.max_0)
        
        final_df = pd.DataFrame()
        # final_df=df
        
        for agent in agent_dict.values():
            df_agent = pd.concat([agent.data[t_init:t_end],df[t_init:t_end]],axis=1)
            df_agent = df_agent.set_index(df_agent.index.map(lambda x: (f"{agent.id}",x)))
            # df_agent = df_agent.rename(columns={agent.id: 'load', agent.pv_id: 'gen'})

            final_df = pd.concat([final_df,df_agent])
            
        utilities().print_info('Local fix: Making gen the gen_com by erasing the columns in the df. this must be addressed if we wnat flexibility in the way observation space is defined')
        
        final_df.drop(columns='gen', inplace=True)
        final_df.drop(columns='load_com', inplace=True)
        final_df.rename(columns={'gen_com': 'gen'}, inplace=True)
                
        return final_df
         
    @staticmethod
    def get_limits(data, mode, var):
        """
        returns the max, min values of vars in the community dataset to be used when defining the observation limits in the environment
        """
        
        lims=data.describe()

        return lims.loc[mode][var]
    
    @staticmethod
    def inspect_timeslot(data, timeslot):
        """
        Extracts all records from a MultiIndex DataFrame for a specific timeslot.
    
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame with a MultiIndex, where the second level corresponds to timeslots.
        timeslot : int
            The value of the timeslot to filter by (e.g., an integer, string, or timestamp).
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing all rows from the specified timeslot across all first-level indices.

        """
        return data.loc[(slice(None), timeslot), :]





    

class GecadDataProcessor():
    def __init__(self, 
                 problem_config,
                 agents_config,
                 file):
        
        self.parser=YAMLParser()
        self.problem_config=self.parser.load_yaml(problem_config)
        self.agents_config=self.parser.load_yaml(agents_config)

        # self.data=self.get_raw_data(file)
        self.data=self.get_clean_data(file)
        self.step_size=self.problem_config['step_size']
        
        
        
    def get_agent_data(self, data_ids):
        """
        Extract the agent specific data for each agent
        Data ids must be columns from a pandas dataframe
        """
        agent_data=self.data[data_ids]
        agent_data_copy=agent_data.copy()
        new_column_names = ['minutes', 'load', 'gen']

        # Rename columns using the new column names
        agent_data_copy.rename(columns=dict(zip(agent_data_copy.columns, new_column_names)), inplace=True)

        
        return agent_data_copy
    
    
    def get_clean_data(self,file) -> pd.DataFrame:
        """
        imports the clean gecad file
        """
        return pd.read_csv(file,index_col=0)
        
    def get_raw_data(self,file) -> pd.DataFrame:
        """
        Extracts consumers and producers from gecad original file and adds timestep and mins
        and returns a complete dataset with generation and consumption
        
        In the future it may get other data such as temperature, beahvior, occupation, etc
        
        """
        print('parsing dataset for community class')
        
        dt=15
        
        cons_data=pd.read_excel(file, 'Total Consumers')
        cons_data.columns=['ag'+str(k) for k in range(len(cons_data.columns))]
        
        prod_data=pd.read_excel(file, 'Total Producers')
        prod_data.columns=['PV'+str(k) for k in range(len(prod_data.columns))]
        
        
        #create a vector of minutes
        mins=pd.DataFrame(np.tile(np.linspace(0,1440-dt,num=int((24*60/dt))),366), columns=['minutes'])
        
        if self.problem_config['dataset_unit']=='kwh':
            dh=dt*(1/60.0)#kw to kwh convertion factor
            data=pd.concat([mins,dh*cons_data,dh*prod_data],axis=1) #kwh
        elif self.problem_config['dataset_unit']=='kw':
            data=pd.concat([mins,cons_data,prod_data],axis=1) #original dataset unit kw

        return data

        



class DataPostProcessor:
    def __init__(self, env):
        self.env=env
        self.ds_unit=self.env.com.problem_conf['dataset_unit']
        # self.metrics=Metrics(env)
    
    def get_post_data(self):
        """This function produces dataframes for analysing results from the 
        environment itself:
            
            
        - the environment stores all the state-action history 
        - From there we can recover and post-process data
        
        - This function is used while testing the environment in the TestEnv class
        
        
        """
        
        
        df=pd.DataFrame()
        df_temp=pd.DataFrame()
        T=self.env.Tw
        for aid in self.env.agents_id:
            #we need to take out the last observation because its allways one timestep ahead
            state_hist=self.env.state_hist.loc[aid][0:T] 
            action_hist=self.env.action_hist.loc[aid]    
            reward_hist=self.env.reward_hist.loc[aid]

            
            df=pd.concat([df,
                           pd.concat([state_hist,action_hist, reward_hist],
                        axis=1)])
        
            
        for aid in self.env.agents_id:  
            # df=pd.concat([state_hist,action_hist, reward_hist],axis=1)
            df.loc[aid,'shift']=df.loc[aid]['action']*self.env.com.agents[aid].apps[0].get_profile(self.ds_unit,self.env.tstep_size)[0]
            df.loc[aid,'shift_base']=df.loc[aid]['action']*self.env.com.agents[aid].apps[0].get_profile(self.ds_unit,self.env.tstep_size)[0]+df.loc[aid]['load0'] #for each agent the sum of its shiftable load with its base load
            
        df['shift'].name='shiftable load for each agent'

        #Sum all variables over the agents for each timestep
        df_group=df.groupby('tstep').sum() # In this dataframe al variables are summed
        
        
        
        for aid in self.env.agents_id: 
            df.loc[aid,'shift_T']=df_group['shift'].values #sum of all shiftable loads for all agents
            df.loc[aid,'shift_base_T']=df_group[['shift','load0']].sum(axis=1).values # sum of all shiftable loads and base loads for all agents
            df.loc[aid,'baseload_T']=df_group['load0'].values
            
            
            df.loc[aid,'coef_shift']=df.loc[aid]['shift']/df.loc[aid]['shift_T'] #sharing/load coeficient considering only the shiftable loads
            df.loc[aid,'coef_shift']=df.loc[aid,'coef_shift'].fillna(0)
            

            df.loc[aid,'coef_shift_base']=df.loc[aid]['shift_base']/df.loc[aid]['shift_base_T'] #sharing/load coefficient considering shiftable and base load
            # df=df.infer_objects(copy=False)

            df.loc[aid,'coef_shift_base']=df.loc[aid,'coef_shift_base'].astype(float).fillna(0)

            utilities().print_info('Individual Cost computing: Real individual cost for each agent AFFECTED by the sharing coefficient, i.e, we assume that excess is shared according to the APPLIANCE load level of each agent')
            
            df.loc[aid,'r_cost']=(df.loc[aid]['shift']-df.loc[aid,'coef_shift']*df.loc[aid,'excess0'])*df.loc[aid,'tar_buy']
            
            df.loc[aid,'r_cost_pos']=df.loc[aid,'r_cost'].clip(lower=0)
            
        
        ### transform the results in order to show them based on the same timesteps 
        df_list=[]
        for aid in self.env.agents_id: 
            df_ag=df.loc[aid]
            new_col_names={col_name:col_name+'_'+aid for col_name in df_ag.columns}
            df_ag=df_ag.rename(columns=new_col_names)
            df_ag=df_ag.rename(columns={'tstep_'+aid:'tstep'})
            df_ag=df_ag.set_index('tstep')
            df_list.append(df_ag)
            
        df_select=pd.concat(df_list, axis=1)
        

        #Now we selct only the variables we wanna see
        
        vars_names=['minutes',
                    'gen0',
                    'load0',
                    'excess0',
                    'tar_buy',
                    'action',
                    'reward',
                    'shift',
                    'cost',
                    'baseload_T']
        
        cols_names=[]
        for k in df_select.columns:
            for j in vars_names:
                if j in k:
                    cols_names.append(k)
                    
        
        df_post=df_select[cols_names] #select the variables that are relevant
        df_post=df_post.drop(columns=[k for k in df_post.columns if 'tar_buy0' in k]) #we dont need tar_buy0
        
        #reduce common features and filter out the common variables              
        vars_names_filter=['minutes','shift_T','shift_base_T','excess','load0','tar_buy','baseload_T']
        df_post=self.compare_equals(vars_names_filter, df_post, drop=False) 
        
        
        #compute the Total cost
        df_post_costs=df_post[[k for k in df_post.columns if 'cost_pos' in k]]
        df_post['Cost_shift_T']=df_post_costs.sum(axis=1)

        # compute total reward
        df_post_rewards=df_post[[k for k in df_post.columns if 'reward' in k]] 
        
        if self.env.com.scenarios_conf['game_setup']=='cooperative_colective':#reward is cooperative what means that all agents get the same reward
            print('reward is common for all agents')
            df_post=self.compare_equals(['reward'], df_post, drop=False)
        
            
            
            

                            
       
            
            
        
        
        # # cols_names=[k for k in vars_names if k in df_select.columns]
        
        # columns_names=[]

        # shift_columns_names=['shift_'+aid for aid in menv.agents_id]
        # reward_columns_names=['reward_'+aid for aid in menv.agents_id]
        # load_columns_names=['load_'+aid for aid in menv.agents_id]
        # coef_columns_names=['coef_'+aid for aid in menv.agents_id]
        # real_cost_columns_names=['real_cost_'+aid for aid in menv.agents_id]
        
       
        # columns_names.extend(load_columns_names)
        # columns_names.extend(shift_columns_names)
        # columns_names.extend(reward_columns_names)
        # columns_names.extend(coef_columns_names)
        # columns_names.extend(real_cost_columns_names)
        
        # columns_names.extend(['shift_T','load_T','gen0','excess0','reward_T','Cost_shift_T','tar_buy'])
        
        # #make a new dataframe to store the solutions
        # df_post=pd.DataFrame(columns=columns_names)
        
        # for aid in menv.agents_id:
        #     var_ag=[v for v in df_post.columns if aid in v]
            
        #     for var in var_ag:
        #         if 'load' in var:
        #             df_post[var]=df.loc[aid,'load0'].values
                
        #         if 'shift' in var:
        #             df_post[var]=df.loc[aid,'action'].values*menv.profile[aid][0]
                    
        #         if 'reward' in var:
        #             df_post[var]=df.loc[aid,'reward'].values
                    

                    
        
        
        # df_post['shift_T']=df_post[shift_columns_names].sum(axis=1)
        # # df_post['']
        
        # #we have to perfomr again this cicle
        # for aid in menv.agents_id:
        #     var_ag=[v for v in df_post.columns if aid in v]
        #     for var in var_ag:
        #             if 'coef' in var:
        #                 df_post[var]=df.loc[aid,'action'].values*menv.profile[aid][0]/df_post['shift_T']
        #                 df_post[var]=df_post[var].fillna(0)  #substitute all nans for zeros 
                    
                   
        
        # df_post['load_T']=df_post[load_columns_names].sum(axis=1)
        # df_post['reward_T']=df_post[reward_columns_names].sum(axis=1)
        
        
        # df_post['gen0']=df.loc[menv.agents_id[0],'gen0'].values #pv production is the same for all and it is collective. So we can use any agent on the agents_id list
        # df_post['excess0']=df.loc[menv.agents_id[0],'excess0'].values # the excess is the same for all
        
        
        # df_post['tar_buy']=df.loc[menv.agents_id[0],'tar_buy'].values
        
        
        # #computing cost of ONLY the shiftable loads with excess
        # df_temp=df_post['shift_T']-df_post['excess0']
        # df_post['Cost_shift_T']=np.maximum(df_temp,0)*df_post['tar_buy']
        
        
        

            
        return df, df_post
        

    
    @staticmethod
    def compare_equals(vars_names,df, drop=True):
        """We are loping over the vars_names_filter and check if the columns for each agent are equal. if they are we are droping them and subsituting by only a column with the common variable its applicable to minutes, total shiftable loads and total bse load"""
        
        df=df.copy()
        for n in vars_names:
            names=[k for k in df.columns if n in k]
            vals_var=[]
            for col0 in names:
                for col1 in [na for na in names if na != col0]:
                    # ic(col0)
                    # ic(col1)
                    # ic(df[col0]==df[col1])
                    val=df[col0]==df[col1]
                    # ic(all(val))
                    vals_var.append(all(val))
                    if all(vals_var):
                            df_temp=df[names[0]]
                            df.loc[:,n]=df_temp # bug
                            if drop:
                                df=df.drop(columns=names)
                            # ic(col0)    
                            names.remove(col1)
        return df
        


    def get_episode_metrics(self,full_state,env_state,k):
        agents_id=full_state.index.unique()
        metrics=pd.DataFrame(index=full_state.index.unique())
        Total_load=[]
        
        # import pdb
        # pdb.pdb.set_trace()
        #Per agent metrics
        
        
        for ag in agents_id:
            
            
            
            # per agent cost considering the 
            # full_state.loc[ag,'cost']=(full_state.loc[ag]['action']*environment.profile[ag][0]-full_state.loc[ag]['excess0'])*full_state.loc[ag]['tar_buy']
            
            
            

            # # cost=pd.DataFrame([max(0,k) for k in full_state.loc[ag]['cost']])
            # pos_cost=pd.DataFrame([max(0,k) for k in full_state.loc[ag]['cost']])
            # full_state.loc[ag,'cost_pos']=pos_cost.values        
            
            # Self-Sufficiency
            # full_state.loc[ag,'load']=full_state.loc[ag]['action']*environment.profile[ag][0]
            
            # full_state.loc[ag,'selfsuf']=full_state.loc[ag][['load','excess0']].min(axis=1)/environment.E_prof.loc[ag]['E_prof'] #dh its just here as a patch
            
            
            
            #group everything
            metrics.loc[ag,'cost']=full_state.loc[ag,'r_cost_pos'].sum()
            # metrics.loc[ag,'selfsuf']=full_state.loc[ag,'selfsuf'].sum()
            
            # should per agent x_ratio have the sharing coefficient?
            utilities().print_info('x_ratio is computed with a single agent excess: only works for common excess')
            metrics.loc[ag,'x_ratio']=full_state.loc['ag1']['excess0'].sum()/self.env.agents_params.loc[ag,'E_prof']
            
            metrics.loc[ag,'reward_episode']=full_state.loc['ag1']['reward'].sum()
            
        
        #community metrics
        # SS_temp=pd.concat([full_state[['minutes','load']]\
        #     .set_index('minutes')\
        #     .groupby(level=0).sum(),
        #     full_state.iloc[0:environment.Tw][['minutes','excess0']].set_index('minutes')],axis=1)
        
        # metrics.loc['com','selfsuf']=SS_temp.min(axis=1).sum()/(environment.E_prof['E_prof'].sum()*
        #                               (environment.Tw/environment.tstep_per_day)) #number of days
        
        utilities().print_info('self-sufficiency computed only for shiftable loads')
        metrics.loc['com','selfsuf']=env_state[['shift_T','excess']].min(axis=1).sum()/self.env.agents_params['E_prof'].sum()
        
        #compute the ratio between energy needed and excess available
        # E_ratio=full_state.loc['ag1']['excess0'].sum()/self.env.E_prof['E_prof'].sum()
        
        E_ratio=env_state['excess'].sum()/self.env.agents_params['E_prof'].sum()
        metrics.loc['com','x_ratio']=E_ratio
        
        metrics.loc['com','x_sig']=self.sigmoid(0.5,6.2,2,1,E_ratio)

        
        
        
        # metrics.loc['com','selfsuf']=full_state['selfsuf'].sum()/self.env.num_agents
        # metrics.loc['com','cost']=full_state['cost_pos'].sum()
        metrics.loc['com','cost']=env_state['Cost_shift_T'].sum() #the cost of consuming aggregated shiftable load
        
        
        #Binary metrics
        min_cost=full_state['tar_buy'].min()*self.env.agents_params['E_prof'].sum()
        #1 if the cost is greater that mininmum tarif cost of community
        metrics.loc['com','y']=int(bool(metrics.loc['com','cost'] > min_cost))
        
        #cost variation relative to the min cost (-1 zero cost)
        # min_cost=self.env.tar_buy*self.env.E_prof['E_prof'].sum()
        metrics['cost_var']=(metrics.loc['com']['cost']-min_cost)/min_cost
        
        
        #year season
        metrics['day']=self.env.tstep_init/self.env.tstep_per_day
        metrics['tstep']=self.env.tstep_init
        metrics['season']=self.get_season(metrics.loc['com']['day'])
        
        
        metrics.loc['com','reward_episode']=full_state['reward'].sum()
        #create index for test episode number
        metrics['test_epi']=k
        metrics_out=metrics.set_index('test_epi',drop=True, append=True)
        
        
        #year season

        
        return metrics_out


    def get_season(self,day):
            # "day of year" ranges for the northern hemisphere
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)
        # winter = everything else
        
        if day in spring:
            season = 'spring'
        elif day in summer:
            season = 'summer'
        elif day in fall:
            season = 'fall'
        else:
            season = 'winter'
        
        return season
    

    def sigmoid(self,a,b,c,d,x):
        return c/(d+np.exp(-a*x+b))
    
    def self_suf(self,action):
        # if var > 0 and env.gen0 != 0:
            # g=var
        # elif var <= 0:
            # g=env.load0
        #BUG    
        if self.env.gen0 != 0:
            # g=min(env.load0+action*env.profile[0],env.gen0) #for the total load 
            g=min(action*self.env.com.agents['ag1'].apps[0].get_profile(self.ds_unit,self.env.tstep_size)[0],self.env.excess0) #bug #self sufficency just for the load
        elif self.env.gen0 ==0:
            g=0
        
        return g
            


class OptiDataPostProcessor(DataPostProcessor):
    def __init__(self, env):
        self.env=env
        
    def get_episode_metrics(self, solution, k):
        pass
        
        

class EnvDataProcessor():
    """
    This class only makes sense with environment data outputed from DataProcessor
    """
    def __init__(self, env_data, window):
        self.data=env_data
        self.Tw=window
    
    
    def get_one_day(self,t_start,t_end):
        return self.data.loc[(slice(None), slice(t_start, t_end)), :]
    
    def get_daily_stats(self):
        allowed_inits=self.get_allowed_inits()
        daily_stats=[]
        for ts in allowed_inits:
            one_day=self.get_one_day(ts,ts+self.Tw)
            daily_stats.append(one_day.groupby(level=0).sum())
            
        merged=pd.concat(daily_stats)
        merged=merged.drop(columns=['minutes'])
        stats={}
        agents=self.data.index.levels[0]
        for agent in agents:
            stats[agent]=merged.loc[agent].describe()   
        
        return pd.concat(stats)  
    

    def get_data_stats(self):
        stats={}
        agents=self.data.index.levels[0]
        for agent in agents:
            stats[agent]=self.data.loc[agent].describe()        
        return stats        

    
    def get_allowed_inits(self):
        return self.data[self.data['minutes']==0].index.get_level_values(1).unique().tolist()
        


    def get_selected_allowed_inits(self,config,agents_params):
        """
        returns the initial timeslot for each day, i.e, time=00:00
        removes the day for which the PV is the same during the wholle day to clean out continuous values PV
        """
        
        def detect_anomalies_pv(time_series,k):
            time_series=pd.DataFrame(time_series)
            counts=time_series.value_counts()
            
            # if counts[0.0]==len(time_series): #PV is zero in the wholle day
            if counts[counts==len(time_series)].any():
                return True
            
            counts=counts.drop(0.0, level='gen') #remove the number of zero occurence
            if counts[counts>=k].any():
                return True
            else:
                return False
        
        
        t_inits=self.get_allowed_inits()
        print('init:', len(t_inits))
        
        t_to_remove=[]
        if config['clean_pv_data']:
            for t in t_inits:
                data_day=self.get_one_day(t,t+self.Tw)
                # if data_day.loc['ag1']['gen'].nunique()==1:
                pv=data_day.loc['ag1']['gen']
                if detect_anomalies_pv(pv, 8):
                    t_to_remove.append(t)
        
        t_new = [x for x in t_inits if x not in t_to_remove]
        t_to_remove2=[]
        if config['day_select']:
            for t in t_new:
                data_day=self.get_one_day(t,t+self.Tw)
                # just keep the hard days. Increase threashhold to include more days
                if (data_day.loc['ag1']['gen'].sum()/agents_params['E_prof'].sum()) >= config['pv_threshold']:
                    t_to_remove2.append(t) 
        
        t_inits.sort()
        t_to_remove.sort()       
        t_to_remove2.sort()   

        t_final = [x for x in t_new if x not in t_to_remove2]
        
        t_final.sort()
        
        return t_final
        

    
        
        
        
        
        
        
    
