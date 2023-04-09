#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark all the baseline agents
on a given CyberBattleSim environment and compare
them to the dumb 'random agent' baseline.

NOTE: You can run this `.py`-notebook directly from VSCode.
You can also generate a traditional Jupyter Notebook
using the VSCode command `Export Currenty Python File As Jupyter Notebook`.
"""

# pylint: disable=invalid-name


# In[3]:


import sys
import logging
import gym
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_randomcredlookup as rca
import cyberbattle.agents.baseline.agent_tabularqlearning as tqa
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_ddql as ddqla
import cyberbattle.agents.baseline.agent_dueling_dql as duelingdqla
import cyberbattle.agents.baseline.agent_dueling_ddql as dueling_ddqla

from cyberbattle.agents.baseline.agent_wrapper import Verbosity

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# In[4]:


# Papermill notebook parameters

#############
# gymid = 'CyberBattleTiny-v0'
#############
# gymid = "CyberBattleToyCtf-v0"
# env_size = None
# iteration_count = 1500
# training_episode_count = 20e
# eval_episode_count = 10
# maximum_node_count = 12
# maximum_total_credentials = 10
#############
# gymid = "CyberBattleChain-v0"
# env_size = 10
# iteration_count = 9000
# training_episode_count = 50
# eval_episode_count = 5
# maximum_node_count = 22
# maximum_total_credentials = 22

env_size = None

# In[23]:


# Parameters
gymid = "CyberBattleToyCtf-v0"
iteration_count = 2500
training_episode_count = 20
eval_episode_count = 10
maximum_node_count = 12
maximum_total_credentials = 10

# In[24]:


# Load the Gym environment
if env_size:
    gym_env = gym.make(gymid, size=env_size)
else:
    gym_env = gym.make(gymid)

ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=maximum_node_count,
    maximum_total_credentials=maximum_total_credentials,
    identifiers=gym_env.identifiers
)

# In[25]:


debugging = True
if debugging:
    print(f"port_count = {ep.port_count}, property_count = {ep.property_count}")

    gym_env.environment
    # training_env.environment.plot_environment_graph()
    gym_env.environment.network.nodes
    gym_env.action_space
    gym_env.action_space.sample()
    gym_env.observation_space.sample()
    o0 = gym_env.reset()
    o_test, r, d, i = gym_env.step(gym_env.sample_valid_action())
    o0 = gym_env.reset()

    o0.keys()

    fe_example = w.RavelEncoding(ep, [w.Feature_active_node_properties(ep), w.Feature_discovered_node_count(ep)])
    a = w.StateAugmentation(o0)
    w.Feature_discovered_ports(ep).get(a, None)
    fe_example.encode_at(a, 0)

# In[7]:


# In[26]:


# Evaluate the Deep Q-learning agent
dql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="DQL"
)

# In[27]:


# Evaluate an agent that exploits the Q-function learnt above
dql_exploit_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=dql_run['learner'],
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    epsilon_minimum=0.00,
    render=False,
    plot_episodes_length=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting DQL"
)

# In[28]:


# Evaluate the Double Deep Q-learning agent
ddql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=ddqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="DDQL"
)

# In[29]:


# Evaluate an agent that exploits the Double Q-function learnt above
ddql_exploit_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=ddql_run['learner'],
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    epsilon_minimum=0.00,
    render=False,
    plot_episodes_length=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting DDQL"
)

# In[30]:


# Evaluate the Dueling Deep Q-learning agent
dueling_dql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=duelingdqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="Dueling DQL"
)

# In[31]:


# Evaluate an agent that exploits the Dueling Q-function learnt above
dueling_dql_exploit_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=dueling_dql_run['learner'],
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    epsilon_minimum=0.00,
    render=False,
    plot_episodes_length=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting Dueling DQL"
)

# In[32]:


# Evaluate the Dueling Double Deep Q-learning agent
dueling_ddql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=dueling_ddqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="Dueling DDQL"
)

# In[33]:


# Evaluate an agent that exploits the Dueling Double Q-function learnt above
dueling_ddql_exploit_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=dueling_ddql_run['learner'],
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    epsilon_minimum=0.00,
    render=False,
    plot_episodes_length=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting Dueling DDQL"
)



# Evaluate the random agent
random_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=learner.RandomPolicy(),
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=1.0,  # purely random
    render=False,
    verbosity=Verbosity.Quiet,
    plot_episodes_length=False,
    title="Random search"
)


# In[34]:


# Evaluate a random agent that opportunistically exploits
# credentials gathere in its local cache
credlookup_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=rca.CredentialCacheExploiter(),
    episode_count=10,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_exponential_decay=10000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="Credential lookups (ϵ-greedy)"
)


# In[ ]:


# Evaluate a Tabular Q-learning agent
tabularq_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=tqa.QTabularLearner(
        ep,
        gamma=0.015, learning_rate=0.01, exploit_percentile=100),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.01,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="Tabular Q-learning"
)


# In[9]:


# Evaluate an agent that exploits the Q-table learnt above
tabularq_exploit_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=tqa.QTabularLearner(
        ep,
        trained=tabularq_run['learner'],
        gamma=0.0,
        learning_rate=0.0,
        exploit_percentile=90),
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    render=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting Q-matrix"
)








# In[35]:

#
# # Compare and plot results for all the agents
# all_runs = [
#     random_run,
# #     credlookup_run,
# #     tabularq_run,
# #     tabularq_exploit_run,
#     dql_run,
#     dql_exploit_run,
#         ddql_run,
#     ddql_exploit_run
# ]
#
# # Plot averaged cumulative rewards for DQL vs Random vs DQL-Exploit
# themodel = dqla.CyberBattleStateActionModel(ep)
# p.plot_averaged_cummulative_rewards(
#     all_runs=all_runs,
#     title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n'
#     f'State: {[f.name() for f in themodel.state_space.feature_selection]} '
#     f'({len(themodel.state_space.feature_selection)}\n'
#     f"Action: abstract_action ({themodel.action_space.flat_size()})")


# In[36]:

#
# # Compare and plot results for all the agents
# all_runs = [
#     random_run,
# #     credlookup_run,
# #     tabularq_run,
# #     tabularq_exploit_run,
#     dql_run,
#     dql_exploit_run,
#         ddql_run,
#     ddql_exploit_run,
#     dueling_dql_run,
#     dueling_dql_exploit_run
#
# ]
#
# # Plot averaged cumulative rewards for DQL vs Random vs DQL-Exploit
# themodel = dqla.CyberBattleStateActionModel(ep)
# p.plot_averaged_cummulative_rewards(
#     all_runs=all_runs,
#     title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n'
#     f'State: {[f.name() for f in themodel.state_space.feature_selection]} '
#     f'({len(themodel.state_space.feature_selection)}\n'
#     f"Action: abstract_action ({themodel.action_space.flat_size()})")


# In[37]:
#
#
# # Compare and plot results for all the agents
# all_runs = [
#     random_run,
# #     credlookup_run,
# #     tabularq_run,
# #     tabularq_exploit_run,
#     dql_run,
#     dql_exploit_run,
#         ddql_run,
#     ddql_exploit_run,
# #     dueling_dql_run,
# #     dueling_dql_exploit_run,
#     dueling_ddql_run,
#     dueling_ddql_exploit_run
#
#
# ]
#
# # Plot averaged cumulative rewards for DQL vs Random vs DQL-Exploit
# themodel = dqla.CyberBattleStateActionModel(ep)
# p.plot_averaged_cummulative_rewards(
#     all_runs=all_runs,
#     title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n'
#     f'State: {[f.name() for f in themodel.state_space.feature_selection]} '
#     f'({len(themodel.state_space.feature_selection)}\n'
#     f"Action: abstract_action ({themodel.action_space.flat_size()})")
#
#
# # In[38]:


# Compare and plot results for all the agents
all_runs = [
    random_run,
    credlookup_run,
    tabularq_run,
    tabularq_exploit_run,
    dql_run,
    dql_exploit_run,
    ddql_run,
    ddql_exploit_run,
    dueling_dql_run,
    dueling_dql_exploit_run,
    dueling_ddql_run,
    dueling_ddql_exploit_run

]

# Plot averaged cumulative rewards for DQL vs Random vs DQL-Exploit
themodel = dqla.CyberBattleStateActionModel(ep)
p.plot_averaged_cummulative_rewards(
    all_runs=all_runs,
    title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n'
          f'State: {[f.name() for f in themodel.state_space.feature_selection]} '
          f'({len(themodel.state_space.feature_selection)}\n'
          f"Action: abstract_action ({themodel.action_space.flat_size()})")

# In[13]:

#
# # Compare and plot results for all the agents
# all_runs = [
#     random_run,
#     credlookup_run,
#     tabularq_run,
#     tabularq_exploit_run,
#     dql_run,
#     dql_exploit_run
# ]
#
# # Plot averaged cumulative rewards for DQL vs Random vs DQL-Exploit
# themodel = dqla.CyberBattleStateActionModel(ep)
# p.plot_averaged_cummulative_rewards(
#     all_runs=all_runs,
#     title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n'
#     f'State: {[f.name() for f in themodel.state_space.feature_selection]} '
#     f'({len(themodel.state_space.feature_selection)}\n'
#     f"Action: abstract_action ({themodel.action_space.flat_size()})")
#
#
# # In[14]:


contenders = [
    random_run,
    credlookup_run,
    tabularq_run,
    tabularq_exploit_run,
    dql_run,
    dql_exploit_run,
    ddql_run,
    ddql_exploit_run,
    dueling_dql_run,
    dueling_dql_exploit_run,
    dueling_ddql_run,
    dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders)

# In[15]:


# Plot cumulative rewards for all episodes
for r in contenders:
    p.plot_all_episodes(r)


contenders1 = [
    random_run,
    #     credlookup_run,
    #     tabularq_run,
    #     tabularq_exploit_run,
    dql_run,
    dql_exploit_run,
    ddql_run,
    ddql_exploit_run,
    dueling_dql_run,
    dueling_dql_exploit_run,
    dueling_ddql_run,
    dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders1)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders1)

# In[15]:


# Plot cumulative rewards for all episodes
for r in contenders1:
    p.plot_all_episodes(r)


contenders2 = [
    random_run,
    credlookup_run,
    tabularq_run,
    #     tabularq_exploit_run,
    dql_run,
    # dql_exploit_run,
    ddql_run,
    # ddql_exploit_run,
    dueling_dql_run,
    # dueling_dql_exploit_run,
    dueling_ddql_run,
    # dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders2)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders2)

contenders3 = [
    random_run,
    #     credlookup_run,
    #     tabularq_run,
    tabularq_exploit_run,
    # dql_run,
    dql_exploit_run,
    # ddql_run,
    ddql_exploit_run,
    # dueling_dql_run,
    dueling_dql_exploit_run,
    # dueling_ddql_run,
    dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders3)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders3)

contenders3v2    = [
    # random_run,
    #     credlookup_run,
    #     tabularq_run,
    tabularq_exploit_run,
    # dql_run,
    dql_exploit_run,
    # ddql_run,
    ddql_exploit_run,
    # dueling_dql_run,
    dueling_dql_exploit_run,
    # dueling_ddql_run,
    dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders3v2)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders3v2)

contenders3v3   = [
    # random_run,
    #     credlookup_run,
    #     tabularq_run,
    # tabularq_exploit_run,
    # dql_run,
    dql_exploit_run,
    # ddql_run,
    ddql_exploit_run,
    # dueling_dql_run,
    dueling_dql_exploit_run,
    # dueling_ddql_run,
    dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders3v3)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders3v3)

contenders4 = [
    # random_run,
    credlookup_run,
    tabularq_run,
    # tabularq_exploit_run,
    # dql_run,
    # dql_exploit_run,
    ddql_run,
    # ddql_exploit_run,
    dueling_dql_run,
    # dueling_dql_exploit_run,
    dueling_ddql_run,
    # dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders4)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders4)


contenders5 = [
    # random_run,
    #     credlookup_run,
    #     tabularq_run,
    tabularq_exploit_run,
    # dql_run,
    # dql_exploit_run,
    # ddql_run,
    ddql_exploit_run,
    # dueling_dql_run,
    dueling_dql_exploit_run,
    # dueling_ddql_run,
    dueling_ddql_exploit_run
]
p.plot_episodes_length(contenders5)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
          f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders5)
