#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:59:37 2024

@author: omega
"""

from rl.models.models2 import ActionMaskModel, CCActionMaskModel
from utils.dataprocessor import YAMLParser

from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config

# from shiftenvRLlib_mas import ShiftEnvMas

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
# from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial


# from auxfunctions_shiftenv import *
from termcolor import colored




from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config

# from shiftenvRLlib_mas import ShiftEnvMas

from ray import tune, air, train
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
# from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial


# from auxfunctions_shiftenv import *
from termcolor import colored

#%%

class Experiment():
    def __init__(self,env,config):
        self.env=env
        self.parser=YAMLParser()
        self.config=self.parser.load_yaml(config)
        
        self.exp_name=self.config['exp_name']
        self.parser=YAMLParser()
        



    def get_policies(self):
        
        """
        Pol_type argument determines if there will be shared policy or each agent will have its own policy
        
        """
        pol_type=self.config['pol_type']
        
        config_pol={}
        
        # import pdb
        # pdb.pdb.set_trace()
        # env_cls=self._get_class(self.config['environment_cls'])
        # if not isinstance(self.env, env_cls):
        #     self.env=self.env.env
        
        #which  policy
        if pol_type=='agent_pol':
        
            policies={'pol_'+aid:(None,
                                self.env.observation_space,
                                self.env.action_space,
                                config_pol,) for aid in self.env.agents_id }
            
            policy_function=self.policy_mapping_fn
            
            
        elif pol_type=='shared_pol':
    
            policies={'shared_pol': (None,self.env.observation_space,self.env.action_space,config_pol,)}
            
            policy_function=self.policy_mapping_fn_shared
         
        # print('Policy Type:', colored(pol_type,'red'))
            
        return policies, policy_function    
     
    @staticmethod
    def policy_mapping_fn(agent_id,episode, worker, **kwargs):
        'Policy mapping function'
        return 'pol_' + agent_id
    
    @staticmethod
    def policy_mapping_fn_shared(agent_id, episode, worker, **kwargs):
        'Policy mapping function with shared policy'
        return 'shared_pol' # parameter sharing must return the same policy for any agent

    
    
    
    def make_algo_config(self, config_file):
        """
        
        
        HPO Algorithms:
            
            
        - Grid Search
            The number of trials of an experiment using HPO algo Grid Search 
            will be determined by the way hiperparameters are defined in the
            algorithm config (including seeds). 
            
                - if [val1, val2] then 2 trials will be created
                
        
        
        
        """
        
        def find_keys_with_lists(d, exclude_keys=None):
            if exclude_keys is None:
                exclude_keys = set()

            keys_with_lists = []

            # Traverse through the dictionary
            for key, value in d.items():
                # If the key is in the exclusion list, skip it
                if key in exclude_keys:
                    continue
                
                # If the value is a list, add the key to the result
                if isinstance(value, list):
                    keys_with_lists.append(key)
                # If the value is a dictionary, recursively check it
                elif isinstance(value, dict):
                    keys_with_lists.extend(find_keys_with_lists(value, exclude_keys))
            
            return keys_with_lists
                
        
        new_configs=self.parser.load_yaml(config_file)
        keys=find_keys_with_lists(new_configs, exclude_keys=['model'])
        
        config_algo = PPOConfig().update_from_dict(new_configs)

        for k in keys:
            if self.config['hpo_algo']=='grid_search':
                config_algo[k]=tune.grid_search(new_configs[k])
                
            elif self.config['hpo_algo']=='asha':
                if k == 'lr':
                    config_algo[k]=tune.uniform(new_configs[k][0], new_configs[k][1])
                if k == 'entropy_coeff':
                    config_algo[k]=tune.uniform(new_configs[k][0], new_configs[k][1])
                if k == 'train_batch_size':
                    config_algo[k]=tune.choice(new_configs[k])
                if k == 'vf_clip_param':
                    config_algo[k]=tune.choice(new_configs[k])
                if k=='sgd_minibatch_size':
                    config_algo[k]=tune.randint(new_configs[k][0], new_configs[k][1])
                if k=='seed':
                    config_algo[k]=tune.grid_search(new_configs[k])
                    
                    
                    
        #updates for environemnt 
        config_algo.environment(observation_space=self.env.observation_space,
                           action_space=self.env.action_space,
                           env_config=self.env.env_config)
        
        #multiagent
        policies, policy_function = self.get_policies()
        
        config_algo.multi_agent(policies=policies,
                      policy_mapping_fn=policy_function)
        
        return config_algo
    
    
    def make_tune_config(self):
        if self.config['hpo_algo']=='asha':
            sched = ASHAScheduler(
                time_attr=self.config['asha_params']['time_attribute'],
                max_t=self.config['asha_params']['max_time'],
                grace_period=self.config['asha_params']['grace'],
                reduction_factor=self.config['asha_params']['red_factor'],)
            
            config_tune=TuneConfig(mode=self.config['mode'],
                                   metric=self.config['metric'],
                                   scheduler=sched,
                                   num_samples=self.config['asha_params']['n_samples'])
        else:
            sched=[]
            config_tune=TuneConfig(mode=self.config['mode'],
                                   metric=self.config['metric'],)
        
        
        return config_tune
    
    def make_run_config(self, results_dir):
        conf=train.RunConfig(verbose=self.config['verbose'], 
                                  name=self.exp_name,
                                  local_dir=results_dir,
                                  storage_path=results_dir)

        return conf

    def _get_class(self, class_name):
        """Retrieve a class object by its name."""
        return globals().get(class_name)
    
    
    
    def get_resources(self):
        num_cpu_head=self.config['num_cpu_head']
        num_gpu_head=self.config['num_gpu_head']
        
        num_cpu_node=self.config['num_cpu_node']
        num_gpu_node=self.config['num_gpu_node']
        n_factor=self.config['n_factor']
        
        # b={'CPU': num_cpu,'GPU': num_gpu}
        
        # resources=tune.PlacementGroupFactory([{'CPU': 1.0}]+ [b]*cpu_factor)
        resources=tune.PlacementGroupFactory([{'CPU': num_cpu_head, 'GPU': num_gpu_head}] + [{'CPU': num_cpu_node, 'GPU': num_gpu_node}] * n_factor)
        
        # resources={'CPU': self.config['num_cpu_head'], 'GPU': self.config['num_gpu_head']}
        
        return resources
        
        # resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': num_cpu}] * 6)

        