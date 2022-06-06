# In this file we implement some helpers functions and algorithms dedicated to our tasks

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from nim_env import NimEnv, OptimalPlayer, QL_Player
import WarningFunctions as wf
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=11)
plt.rc('axes',titlesize=20, labelsize = 15)
plt.rc('legend',fontsize=11)
plt.rc('figure',titlesize=20)

# if gpu is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------ helpers for Q-learning ----------------
def QL_one_game(playerQL, playerOpt, alpha, gamma, env, update = True):
    """
    Implementation of one game of NIM between a Q-learning player (after: QL player) and an optimal player.
    Input:
        - playerQL: an instance of the PlayerQL class
        - playerOpt: an instance of the Optimal player class
        - alpha: learning rate of the QL player
        - gamma: discount factor of the QL player
        - env: an instance of the class NimEnv. Setting with which the players are going to play
        - update: if set to false, the Q-values are not updated. Default: True. 
            The utility of this argument is to be able to play a game without having to update the 
            Q-values (useful when computing Mopt and Mrand)
    Output:
        - reward of the QL-player
    The idea of the update is to keep copies of the environment at different times: 
        - before the turn of the QL player
        - after the turn of the QL player
        - after the turn of the Opt player
    """
    heaps, _, _ = env.observe()
    i = 0
    while not env.end:
        if env.current_player == playerOpt.player:
            Opt_move = playerOpt.act(heaps)
            heaps, end, winner = env.step(Opt_move)
            heaps_after_opt_plays = heaps.copy()
            if i > 0 and update == True:
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL_move, env_after_QLplays = heaps_afterQL_plays, 
                                    env_after_other_plays = heaps_after_opt_plays, env = env, alpha = alpha, gamma = gamma)
        else:
            heaps_beforeQL_move = heaps.copy()    
            move = playerQL.act(heaps)
            heaps, end, winner = env.step(move)
            heaps_afterQL_plays = heaps.copy()
            if env.end and update == True: # otherwise when QL wins, its Q-values are not updated
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL_move, env_after_QLplays = heaps_afterQL_plays, 
                                    env_after_other_plays = heaps_beforeQL_move, env = env, alpha = alpha, gamma = gamma)
        i += 1
    
    return env.reward(playerQL.player)

def QL_one_game_vs_self(playerQL, alpha, gamma, env, update = True):
    """
    Implementation of one game of NIM of a Q-learning player (after: QL player) against itself.
    - inputs:
        - playerQL: an instance of the PlayerQL class. The idea is to then create two copies of this player 
            that will play against each other. Q-values are updated after each game for every instance 
            (the copies and the original) if update is set to True (see after).
        - alpha: learning rate of the QL player
        - gamma: discount factor of the QL player
        - env: an instance of the class NimEnv. Setting with which the players are going to play
        - update: if set to false, the Q-values are not updated. Default: True. 
            The utility of this argument is to be able to play a game without having to update the 
            Q-values (useful when computing Mopt and Mrand)
    - output: None
    
    The idea of the update is to keep a copy of the environment before the turn of a player and after the turn of the other (heaps before and heaps_after) and also the actions played by each.
    If the game is over we need to update both the players'q-values as they both get a reward (-1 or +1).
    """
    playerQL1, playerQL2 = playerQL.copy(), playerQL.copy()
    playerQL1.player = 1
    playerQL2.player = 0
    heaps, _, _ = env.observe()
    i = 0
    while not env.end:
        if env.current_player == playerQL1.player:
            heaps_beforeQL1 = heaps.copy()
            move1 = playerQL1.act(heaps)
            heaps, end, winner = env.step(move1)
            heaps_after_QL1_plays = heaps.copy()
            
        else:
            heaps_beforeQL2 = heaps.copy()
            move2 = playerQL2.act(heaps)
            heaps, end, winner = env.step(move2)
            heaps_after_QL2_plays = heaps.copy()
        if (i > 0 and update == True and (not (env.end))):
            if env.current_player == 0: # player 1 just played
                playerQL.player = 0
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL2, env_after_QLplays = heaps_after_QL2_plays, 
                                    env_after_other_plays = heaps_after_QL1_plays, env = env, alpha = alpha, gamma = gamma)
            else: # player 2 just played
                playerQL.player = 1
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL1, env_after_QLplays = heaps_after_QL1_plays, 
                                    env_after_other_plays = heaps_after_QL2_plays, env = env, alpha = alpha, gamma = gamma)
            
        if env.end and update == True: # we update Q-values for both players
            if env.current_player == 0: # player 1 has won
                playerQL.player = 0 
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL2, env_after_QLplays = heaps_after_QL2_plays, 
                                    env_after_other_plays = heaps_after_QL1_plays, env = env, alpha = alpha, gamma = gamma)
                playerQL.player = 1
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL1, env_after_QLplays = heaps_after_QL1_plays, 
                                    env_after_other_plays = heaps_after_QL2_plays, env = env, alpha = alpha, gamma = gamma)
            else: # player 2 won the game
                playerQL.player = 1
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL1, env_after_QLplays = heaps_after_QL1_plays, 
                                    env_after_other_plays = heaps_after_QL2_plays, env = env, alpha = alpha, gamma = gamma)
                playerQL.player = 0 
                playerQL.update_qval(env_before_QLplays = heaps_beforeQL2, env_after_QLplays = heaps_after_QL2_plays, 
                                    env_after_other_plays = heaps_after_QL1_plays, env = env, alpha = alpha, gamma = gamma)
        
        playerQL1.qvals = playerQL.qvals.copy()
        playerQL2.qvals = playerQL.qvals.copy()
        i += 1
                
def Q1(nb_games = 20000, eps = 0.1, eps_opt = 0.5, alpha = 0.1, gamma = 0.99, step = 250, 
       seed = None, question = 'q2-1', nb_samples = 5, save = True):
    """
    Implements the solution to the 1st question
    - inputs: 
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps: espilon associated to the QL-player (probability of playing at random). Default: 0.1, dtype: float
        - eps_opt: epsilon associated to the optimal player. Default: 0.5, dtype: float
        - alpha: learning rate of the QL-player. Default: 0.1, dtype: float
        - gamma: discount factor of the QL-player. Default: 0.99, dtype: float
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-1', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a plot representing the average reward every 'step' games for the QL-player. According to the value of the argument 'nb_samples', two different plots can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the list of rewards and the time to achieve 80% of final performance (in s)
    """
    
    # Call the warning function to prevent wrong usage
    wf.Q1_warning(nb_games, eps, eps_opt, alpha, gamma, step, question, nb_samples, save)
    
    plt.figure(figsize = (9, 8))
    Rewards = np.zeros(int(nb_games / step))
    Steps = np.zeros(int(nb_games / step))
    Times = np.zeros(int(nb_games / step))
    for s in range(nb_samples):
        total_reward = 0.0
        env = NimEnv(seed = seed)
        playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
        playerQL = QL_Player(epsilon = eps, player = 1)
        time_start = time.time()
        for i in range(nb_games):
            # switch turns at every game
            if i % 2 == 0:
                playerOpt.player = 0
                playerQL.player = 1
            else:
                playerOpt.player = 1
                playerQL.player = 0
        
            total_reward += QL_one_game(playerQL, playerOpt, alpha = alpha, gamma = gamma, env = env)
            if i % step == step - 1:
                Times[i // step] += time.time() - time_start
                Rewards[i // step] += total_reward / step
                Steps[i // step] = i
                total_reward = 0.0
                time_start = time.time()
            env.reset(seed = seed)
    Rewards = Rewards / nb_samples
    Times = Times / nb_samples
    plt.plot(Steps, Rewards)
    plt.title('Evolution of average reward every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for QL-player (' + str(eps) + ')')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
    index_80 = np.argmax(Rewards > 0.8 * Rewards[-1]) + 1
    training_time = np.sum(Times[:index_80])
    return Rewards, training_time

def Q2(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, 
       step = 250, seed = None, question = 'q2-2', nb_samples = 5, save = True):
    """
    Implements the solution to the 2nd question
    - inputs: 
        - N_star: a list containing the values of n*, dtype: list
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - alpha: learning rate of the QL-player. Default: 0.1, dtype: float
        - gamma: discount factor of the QL-player. Default: 0.99, dtype: float
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-3', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with a plot for each n* representing the average reward every 'step' games for the QL-player. According to the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final rewards for each n* as a dictionnary
        - returns the time to achieve 80% of final performance (in s) for each n* as a dictionnary
    """
    wf.Q2_warning(N_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save)
    
    plt.figure(figsize = (9, 8))
    legend = []
    Final_rewards = {}
    training_times = {}
    for j, n_star in enumerate(N_star):
        Times = np.zeros(int(nb_games / step))
        Rewards = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        for s in range(nb_samples):
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star))
            playerOpt = OptimalPlayer(epsilon = 0.5, player = 0)
            playerQL = QL_Player(epsilon = eps, player = 1)
            total_reward = 0.0
            time_start = time.time()
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerQL.player = 1
                else:
                    playerOpt.player = 1
                    playerQL.player = 0
        
                total_reward += QL_one_game(playerQL, playerOpt, alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Times[i // step] += time.time() - time_start
                    Rewards[i // step] += total_reward / step
                    total_reward = 0.
                    Steps[i // step] = i
                    time_start = time.time()
                env.reset(seed = seed)
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        Rewards = Rewards / nb_samples
        Times = Times / nb_samples
        plt.plot(Steps, Rewards)
        Final_rewards['{}'.format(n_star)] = Rewards[-1]
        legend.append(r"$n_* = {}$".format(n_star))
        index_80 = np.argmax(Rewards > 0.8 * Rewards[-1]) + 1
        training_time = np.sum(Times[:index_80])
        training_times[f'{n_star}'] = training_time
    plt.legend(legend)
    plt.title('Evolution of average reward with decrease of exploration level')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for QL-player')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
    return Final_rewards, training_times
    
def Q3(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, 
       step = 250, seed = None, question = 'q2-3', nb_samples = 5, save = True):
    """
    Implements the solution to the 3rd question
    - inputs: 
        - N_star: a list or array containing the values of n*. dtype: int 
        - nb_games: the number of games to play. Default: 20000. dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - alpha: learning rate of the QL-player. Default: 0.1, dtype: float
        - gamma: discount factor of the QL-player. Default: 0.99, dtype: float
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-3', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. According to the value of the argument 'nb_samples', two figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each n* as two dictionnaries
        - returns the time to achieve 80% of final performance of Mopt and Mrand (in s) as two dictionnaries 
            with the values n* as entries
    """
    N_star = list(N_star)
    wf.Q3_warning(N_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save)
    
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    training_times_opt = {}
    training_times_rand = {}
    for j, n_star in enumerate(N_star):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        Times = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            total_reward = 0.0
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star)) 
            playerOpt = OptimalPlayer(epsilon = 0.5, player = 0)
            playerQL = QL_Player(epsilon = eps, player = 1)
            time_start = time.time()
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerQL.player = 1
                else:
                    playerOpt.player = 1
                    playerQL.player = 0
        
                total_reward += QL_one_game(playerQL, playerOpt, alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Times[i // step] += time.time() - time_start
                    Steps[i // step] = i
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv(seed = seed)
                    # save eps
                    eps = playerQL.epsilon
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        playerQL.epsilon = 0
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, new_playerOpt, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset(seed = seed)   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerQL.player = 1
                            else:
                                playerRand.player = 1
                                playerQL.player = 0
                            mrand += QL_one_game(playerQL, playerRand, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset(seed = seed)
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                    time_start = time.time()
                    playerQL.epsilon = eps
                env.reset()
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$n_* = {}$".format(n_star))
        Final_Mopt["{}".format(n_star)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(n_star)] = Mrand[-1] / nb_samples
        Times = Times / nb_samples
        index_80_opt = np.argmax(Mopt > 0.8 * Mopt[-1]) + 1
        index_80_rand = np.argmax(Mrand > 0.8 * Mrand[-1]) + 1
        training_time_opt = np.sum(Times[:index_80_opt])
        training_time_rand = np.sum(Times[:index_80_rand])
        training_times_opt[f'{n_star}'] = training_time_opt
        training_times_rand[f'{n_star}'] = training_times_rand
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n*')
    ax2.set_title('Evolution of Mrand for different n*')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
    return Final_Mopt, Final_Mrand, training_times_opt, training_times_rand
        
def Q4(Eps_opt, n_star = 1000, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, 
       step = 250, seed = None, question = 'q2-4', nb_samples = 5, save = True):
    
    """
    Implements the solution to the 4th question
    - inputs: 
        - Eps_opt: a list containing the values of different eps_opt for the optimal player. dtype: list of floats
        - n_star: the value of n* found in the previous questions to be the best for the QL-player. Default: 1000, dtype: int
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - alpha: learning rate of the QL-player. Default: 0.1, dtype: float
        - gamma: discount factor of the QL-player. Default: 0.99, dtype: float
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-4', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each eps_opt as two dictionnaries
        - returns the time to achieve 80% of final performance of Mopt and Mrand (in s) as two dictionnaries 
            with the values eps_opt as entries
    """
    Eps_opt = list(Eps_opt)
    wf.Q4_warning(Eps_opt, n_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save)
    
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    training_times_opt = {}
    training_times_rand = {}
    for j, eps_opt in enumerate(Eps_opt):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        Times = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star)) 
            playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
            playerQL = QL_Player(epsilon = eps, player = 1)
            time_start = time.time()
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerQL.player = 1
                else:
                    playerOpt.player = 1
                    playerQL.player = 0
        
                reward = QL_one_game(playerQL, playerOpt, alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Times[i // step] += time.time() - time_start
                    Steps[i // step] = i
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    # save eps
                    eps = playerQL.epsilon
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        playerQL.epsilon = 0
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, new_playerOpt, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset() 
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerQL.player = 1
                            else:
                                playerRand.player = 1
                                playerQL.player = 0
                            mrand += QL_one_game(playerQL, playerRand, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                    time_start = time.time()
                    playerQL.epsilon = eps
                env.reset()
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$\varepsilon_o = {}$".format(eps_opt))
        Final_Mopt["{}".format(eps_opt)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(eps_opt)] = Mrand[-1] / nb_samples
        Times = Times / nb_samples
        index_80_opt = np.argmax(Mopt > 0.8 * Mopt[-1]) + 1
        index_80_rand = np.argmax(Mrand > 0.8 * Mrand[-1]) + 1
        training_time_opt = np.sum(Times[:index_80_opt])
        training_time_rand = np.sum(Times[:index_80_rand])
        training_times_opt[f'{eps_opt}'] = training_time_opt
        training_times_rand[f'{eps_opt}'] = training_times_rand
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different optimal epsilons')
    ax2.set_title('Evolution of Mrand for different optimal epsilons')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
    return Final_Mopt, Final_Mrand, training_times_opt, training_times_rand
             
def Q7(Eps, nb_games = 20000, alpha = 0.1, gamma = 0.99, step = 250, seed = None, question = 'q2-7', nb_samples = 5, save = True):
    """
    Implements the solution to the 7th question. The QL-player is trained by playing against itself.
    - inputs: 
        - Eps: a list containing the values of different eps for the QL player. dtype: list of float
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - alpha: learning rate of the QL-player. Default: 0.1, dtype: float
        - gamma: discount factor of the QL-player. Default: 0.99, dtype: float
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-7', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each eps as two dictionnaries
        - - returns the time to achieve 80% of final performance of Mopt and Mrand (in s) as two dictionnaries 
            with the values eps as entries
    """
    Eps = list(Eps)
    wf.Q7_warning(Eps, nb_games, alpha, gamma, step, question, nb_samples, save)
    
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    training_times_opt = {}
    training_times_rand = {}
    for j, eps in enumerate(Eps):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        Times = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            env = NimEnv(seed = seed)
            playerQL = QL_Player(epsilon = eps, player = 0)
            time_start = time.time()
            for i in range(nb_games):
                QL_one_game_vs_self(playerQL, alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Steps[i // step] = i
                    Times[i // step] += time.time() - time_start
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    # save eps
                    eps = playerQL.epsilon
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        playerQL.epsilon = 0
                        playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, playerOpt, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerQL.player = 1
                            else:
                                playerRand.player = 1
                                playerQL.player = 0
                            mrand += QL_one_game(playerQL, playerRand, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                    time_start = time.time()
                    playerQL.epsilon = eps
                
                env.reset(seed = seed)
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$\varepsilon = {}$".format(eps))
        Final_Mopt["{}".format(eps)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(eps)] = Mrand[-1] / nb_samples
        Times = Times / nb_samples
        index_80_opt = np.argmax(Mopt > 0.8 * Mopt[-1]) + 1
        index_80_rand = np.argmax(Mrand > 0.8 * Mrand[-1]) + 1
        training_time_opt = np.sum(Times[:index_80_opt])
        training_time_rand = np.sum(Times[:index_80_rand])
        training_times_opt[f'{eps}'] = training_time_opt
        training_times_rand[f'{eps}'] = training_times_rand
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different epsilon')
    ax2.set_title('Evolution of Mrand for different epsilon')
    ax.set_xlabel('Number of games played against itself')
    ax2.set_xlabel('Number of games played against itself')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
    return Final_Mopt, Final_Mrand, training_times_opt, training_times_rand
              
def Q8(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, step = 250, 
       seed = None, question = 'q2-8', nb_samples = 5, save = True):
    """
    Implements the solution to the 8th question. The QL-player is trained by playing against itself.
    - inputs: 
        - N_star: a list containing the values of different n*. dtype: list of int
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - alpha: learning rate of the QL-player. Default: 0.1, dtype: float
        - gamma: discount factor of the QL-player. Default: 0.99, dtype: float
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-7', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each n* as two dictionnaries 
        - returns the set of Q-values of the QL-player after all games are played for each n* as a dictionnary
    """
    N_star = list(N_star)
    wf.Q8_warning(N_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save)
    
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mrand, Final_Mopt, Final_qvals = {}, {}, {}
    for j, n_star in enumerate(N_star):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star))
            playerQL = QL_Player(epsilon = eps, player = 0)
            
            for i in range(nb_games):
                QL_one_game_vs_self(playerQL, alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    # save eps
                    eps = playerQL.epsilon
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        playerQL.epsilon = 0
                        playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, playerOpt, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerQL.player = 1
                            else:
                                playerRand.player = 1
                                playerQL.player = 0
                            mrand += QL_one_game(playerQL, playerRand, alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                    playerQL.epsilon = eps
                env.reset()
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        Final_Mopt['{}'.format(n_star)] = Mopt[-1] / nb_samples
        Final_Mrand['{}'.format(n_star)] = Mrand[-1] / nb_samples
        Final_qvals['{}'.format(n_star)] = playerQL.qvals
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$n_* = {}$".format(n_star))
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n*')
    ax2.set_title('Evolution of Mrand for different n*')
    ax.set_xlabel('Number of games played against itself')
    ax2.set_xlabel('Number of games played against itself')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if nb_samples > 1:
        plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
    else:
        plt.savefig('./Data/' + question + '.png')
    return Final_Mopt, Final_Mrand, Final_qvals
            
def Q10(qval, configs = ['300', '120', '032'], question = 'q2-10', save = True):

    """
    Implements the solution of the 19th question. 
    inputs: 
        - playerDQN : the agent who predicts the q-values. dtype : DQN_Player
        - configs : array of dimension 3 x 3, representing 3 heaps. The q-values will be predicted from these heaps. 
                    Default: np.array([[3, 0, 0], [1, 2, 0], [0, 3, 2]])
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q3-19', dtype: str
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - output: 
        - a figure with 3 subplots representing the q-values predicted by the DQN-player, for each heap.
    """
    first_config = configs[0]
    second_config = configs[1]
    third_config = configs[2]
    qvals1, qvals2, qvals3 = np.zeros((3, 7)) + float('-inf'), np.zeros((3, 7)) + float('-inf'), np.zeros((3, 7)) + float('-inf')
    for i, future_config in enumerate(qval[first_config]):
        action1, action2, action3 = str(int(first_config[0]) - int(future_config[0])) + str(int(first_config[1]) - int(future_config[1])) + str(int(first_config[2]) - int(future_config[2]))
        action = np.array([int(action1), int(action2), int(action3)])
        non_zero_index = np.argmax(action)
        number_to_take = int(action[non_zero_index])
        qvals1[non_zero_index, number_to_take - 1] = qval[first_config][future_config]

    for i, future_config in enumerate(qval[second_config]):
        action1, action2, action3 = str(int(second_config[0]) - int(future_config[0])) + str(int(second_config[1]) - int(future_config[1])) + str(int(second_config[2]) - int(future_config[2]))
        action = np.array([int(action1), int(action2), int(action3)])
        non_zero_index = np.argmax(action)
        number_to_take = int(action[non_zero_index])
        qvals2[non_zero_index, number_to_take - 1] = qval[second_config][future_config]

    for i, future_config in enumerate(qval[third_config]):
        action1, action2, action3 = str(int(third_config[0]) - int(future_config[0])) + str(int(third_config[1]) - int(future_config[1])) + str(int(third_config[2]) - int(future_config[2]))
        action = np.array([int(action1), int(action2), int(action3)])
        non_zero_index = np.argmax(action)
        number_to_take = int(action[non_zero_index])
        qvals3[non_zero_index, number_to_take - 1] = qval[third_config][future_config]

        
    fig, axs = plt.subplots(3, 1, figsize = (30, 10))
    fig.subplots_adjust(hspace=0.5)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    x_label_list = np.arange(1, 8, 1)
    y_label_list = [1, 2, 3]

    ax1.set_xticks(np.arange(0, 7, 1))
    ax1.set_xticklabels(x_label_list)
    ax1.set_yticks(np.arange(0, 3, 1))
    ax1.set_yticklabels(y_label_list)
    
    ax1.set_xlabel('Number of sticks')
    ax1.set_ylabel('Heap')
    divider1 = make_axes_locatable(ax1)
    ax1.set_title('Current configuration: ' + str(first_config[0]) + ' | ' + str(first_config[1]) + ' | ' + str(first_config[2]))
    im1 = ax1.imshow(qvals1, cmap = 'plasma_r')
    #color bar on the right
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax = cax1, label = 'Q-values')
    
    
    ax2.set_yticks(np.arange(0, 3, 1))
    ax2.set_yticklabels(y_label_list)
    ax2.set_xticks(np.arange(0, 7, 1))
    ax2.set_xticklabels(x_label_list)
    ax2.set_xlabel('Number of sticks')
    ax2.set_ylabel('Heap')
    divider2 = make_axes_locatable(ax2)
    ax2.set_title('Current configuration: ' + str(second_config[0]) + ' | ' + str(second_config[1]) + ' | ' + str(second_config[2]))
    im2 = ax2.imshow(qvals2, cmap = 'plasma_r')
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax = cax2, label = 'Q-values')

    ax3.set_yticks(np.arange(0, 3, 1))
    ax3.set_yticklabels(y_label_list)
    ax3.set_xticks(np.arange(0, 7, 1))
    ax3.set_xticklabels(x_label_list)
    ax3.set_xlabel('Number of sticks')
    ax3.set_ylabel('Heap')
    divider3 = make_axes_locatable(ax3)
    ax3.set_title('Current configuration: ' + str(third_config[0]) + ' | ' + str(third_config[1]) + ' | ' + str(third_config[2]))
    im3 = ax3.imshow(qvals3, cmap = 'plasma_r')
    cax3 = divider3.append_axes("right", size="5%", pad=0.05) 
    plt.colorbar(im3, cax = cax3, label = 'Q-values')

    if save:
        fig.savefig('./Data/' + question + '.png')
    
    

    
#------------- helpers for DQN ----------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
#next state here consists of the next state the DQN player can play; i.e. not the next state that the game has, but the one after. 

class ReplayMemory(object):
    """Replay buffer. """
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition. """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample random batch_size saved transitions. """
        random.seed()
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the numer of transitions inside the memory. """
        return len(self.memory)

def to_input(heaps):
    """change the format of the heaps so that it can be used as an input for the neural network, i.e. converts it to a array of size 9 (binary numbers)"""
    init_state = torch.zeros(9, device = device)
    for i in range(3):
        state = bin(heaps[i])[2:]
        j = 0 
        while j < len(state):
            init_state[i*3 + 2 - j] = np.int16(state[len(state) - 1 - j])
            j += 1
    return init_state.clone().detach()

class DQN(nn.Module):
    """Neural Network. """
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 21)

    def forward(self, x):
        x = (x.to(device))
        x = x.view(-1, 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN_Player(OptimalPlayer):
    def __init__(self, player, policy_net : DQN, target_net : DQN, memory : ReplayMemory, EPS_GREEDY : float = 0.1,
                 GAMMA : float = 0.99, buffer_size : int = 10000, BATCH_SIZE : int = 64, TARGET_UPDATE : int = 500):
        """
        Implements the DQN player for part 3 "Deep Q-Learning".
        inputs: 
        - player : 0 or 1, indicates the player's turn
        - policy_net : the policy network of the DQN player. dtype : DQN
        - target_net : the target network of the DQN player. dtype : DQN
        - memory : memory of the DQN player. dtype : Replaymemory
        - EPS_GREEDY: espilon associated to the DQN-player (probability of playing at random, i.e. epsilon-greedy) Default: 0.1, dtype: float
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size. Default: 10 000, dtype: int
        - BATCH_SIZE : batch size of the DQN player. Default: 64, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        """
        super(DQN_Player, self).__init__(player = player)
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.player = player
        self.EPS_GREEDY = EPS_GREEDY
        self.GAMMA = GAMMA
        self.buffer_size = buffer_size
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_UPDATE = TARGET_UPDATE
        self.count = 0 #to count when to update target_net.

        self.optimizer = optim.Adam(policy_net.parameters(), lr = 1e-4)

    def copy(self):
        """Returns a new DQN_player with the same parameters as the DQN player itself."""
        policy_net2 = DQN().to(device)
        target_net2 = DQN().to(device)
        policy_net2.load_state_dict(self.policy_net.state_dict())
        target_net2.load_state_dict(self.policy_net.state_dict())
        memory2 = ReplayMemory(self.buffer_size)

        new_player = DQN_Player(self.player, policy_net2, target_net2, memory2, 
                                EPS_GREEDY = self.EPS_GREEDY, GAMMA = self.GAMMA, buffer_size = self.buffer_size, 
                                BATCH_SIZE = self.BATCH_SIZE, TARGET_UPDATE = self.TARGET_UPDATE)
        return new_player

        
    def QL_Move(self, heaps):
        """ Given heaps, implements the DQN player choosing an action with epsilon greedy.
        input: 
            - heaps : array of size 3, with the number of sticks in each heap.
        output: 
            - result : action that the DQN chooses, with epsilon greedy. 
                        Array of size 2 : result[0] is the number of the heap, in {1,2,3}, and result[1]
                        is the number of sticks to take, in {1, 2, ..., 7}."""
        state = to_input(heaps)
        random.seed()
        sample = random.random()

        #espilon greedy :
        if sample > self.EPS_GREEDY: 
            with torch.no_grad():
                q = self.policy_net(state) #size 21
                #pick the action with the highest reward
                argmax = torch.argmax(q)
                result = torch.tensor([argmax.div(7, rounding_mode="floor")+1, torch.remainder(argmax, 7)+1], device = device)
                return result
        else:
            #available heaps
            H = torch.tensor([False, False, False])
            H[torch.tensor(heaps) > 0] = True
            #choose a random heap (which is available)
            random.seed()
            h =random.choice(torch.arange(1,4)[H])
            N_max = heaps[h-1]
            #choose a random number of sticks
            random.seed()
            n = random.choice(torch.arange(1,N_max + 1))
            result = torch.tensor([h, n], device = device)
            return result
           
    def act(self, heaps, **kwargs):
        """ Implements a move, given some heaps. """
        return self.QL_Move(heaps)
    
    def predict(self, heaps):
        """ predict q values for some given heaps
        input : 
            - heaps
        output : 
            - q-values for each possible action
        """
        state = to_input(heaps)
        q = self.policy_net(state)
        return q.view(3, -1)

    def save_net(self, question : str = 'q3-0'):
        """
        Save the policy net under name the string "question".
        input : 
            - question. Default: 'q3-0', dtype: str
        """
        PATH = './Data/models/' + question + '.pth'
        torch.save(self.policy_net.state_dict(), PATH)

    def optimize(self):
        """ Optimization part of the DQN algorithm. """
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        #This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))    

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        if [s for s in batch.next_state if s is not None] : #is false if the list is empty
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        else : #if there is no non-final next states
            non_final_next_states = torch.empty(0) 
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch) # size : BATCH_SIZE x 21
        state_action_values = state_action_values.gather(1, ((action_batch[::2]-1)*7+(action_batch[1::2]-1)).view(self.BATCH_SIZE, 1)) # size : BATCH_SIZE x 1

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        if len(non_final_next_states) > 0 :
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() 
            # max(1) : take the maximum per batch on the 21 possibilities. [0]: take the max and not the argmax.
                
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  # size : BATCH_SIZE

        # Compute Huber loss
        criterion = nn.HuberLoss()
        loss = criterion(state_action_values.squeeze(), expected_state_action_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target(self):
        """ Count the number of games played by the DQN player and update the target_net after TARGET_UPDATE games."""
        self.count += 1
        if self.count == self.TARGET_UPDATE:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.count = 0

    def memory_push(self, state, action, next_state, reward): 
        """ Push the transition in the memory buffer. """
        self.memory.push(state, action, next_state, reward)

def DQN_one_game(playerDQN : DQN_Player, playerOpt : OptimalPlayer, env : NimEnv, update : bool = True): 
    """ One game is played in between playerDQN and playerOpt, in the Nim environment env. 
        input: 
            - playerDQN : DQN player who plays. dtype: DQN_Player
            - playerOpt : optimal player who plays against the DQN player. dtype: OptimalPlayer
            - env : the Nim environment inside which the two players play. dtype: NimEnv
            - update : if True, the DQN player learns how to play. Otherwise, the DQN player plays without learning. Default: True, dtype: bool.
        output: 
            - reward : DQN player's reward at the end of the game
            - loss : loss of DQN player's optimization at the end of the game
    """
    heaps, _, _ = env.observe()
    loss = None
    j = 0
    reward = torch.tensor([-1], device=device)
    while not env.end:
        if env.current_player == playerOpt.player:
            move = playerOpt.act(heaps)
            heaps, _, _ = env.step(move)
            if j > 0 :
                # Store the transition in memory
                reward = torch.tensor([env.reward(player = playerDQN.player)], device=device)
                if update == True:
                    next_state = to_input(heaps)
                    playerDQN.memory_push(state_DQN, move_DQN, next_state = next_state, reward = reward)
                    loss = playerDQN.optimize()
        else: 
            move_DQN = playerDQN.act(heaps)
            state_DQN = to_input(heaps)
            is_available = env.check_valid(move_DQN)
            if not is_available :
                #if the action is not valid, we give the agent a negative reward
                reward = torch.tensor([-1], device=device)
                if update == True :
                    next_state = None
                    playerDQN.memory_push(state_DQN, move_DQN, next_state, reward)
                    loss = playerDQN.optimize()
                env.end = True
            else : #if the action is valid, we make a step
                heaps, done, _ = env.step(move_DQN)
                if done : #if the game is finished (done == True), then we give the agent a reward of 1.
                    reward = torch.tensor([1], device=device)
                    if update == True:
                        next_state = None
                        playerDQN.memory_push(state_DQN, move_DQN, next_state, reward)
                        loss = playerDQN.optimize()
              
        j += 1
    if update == True:
        playerDQN.update_target()
    return reward, loss  
    
def DQN_one_game_vs_self(player_DQN : DQN_Player, env : NimEnv, update : bool = True):
    """
    Implementation of one game of NIM of a DQN-learning player (after: DQN player) against itself.
    - inputs:
        - playerDQN: an instance of the PlayerDQN class. Its policy net and its memory are updated after each game for every instance 
            if update is set to True (see after).
        - env: an instance of the class NimEnv. Setting with which the players are going to play
        - update: if set to false, the policy net and the memory are not updated. Default: True. 
    - output: None
    """

    heaps, _, _ = env.observe()
    reward = -1
    loss = None
    i = 0
    while not env.end:
        if env.current_player == 1:
            move_DQN1 = player_DQN.act(heaps)
            state_DQN1 = to_input(heaps)
            is_available = env.check_valid(move_DQN1)
            if not is_available :
                #if the action is not valid, we give the agent a negative reward
                reward = torch.tensor([-1], device=device)
                next_state = None
                if update == True :
                    player_DQN.memory_push(state_DQN1, move_DQN1, next_state, reward)
                    loss = player_DQN.optimize()


                env.end = True
            else : #if the action is valid, we make a step
                heaps, done, _ = env.step(move_DQN1)

                if done : #if the game is finished (done == True), then we give the agent a reward of 1.
                    reward1 = torch.tensor([1], device=device)
                    reward2 = torch.tensor([-1], device=device)
                    next_state = None
                    if update == True:
                        if i>0:
                            player_DQN.memory_push(state_DQN2, move_DQN2, state_DQN1, reward2)
                        player_DQN.memory_push(state_DQN1, move_DQN1, next_state, reward1)
                        loss = player_DQN.optimize()
                else:
                  reward2 = torch.tensor([env.reward(player = 0)], device=device)
                  next_state = to_input(heaps)
                  if i > 0 and update == True:
                    # Store the transition in memory
                    player_DQN.memory_push(state_DQN2, move_DQN2, next_state = next_state, reward = reward2)
                    loss = player_DQN.optimize()
            
        else:
            move_DQN2 = player_DQN.act(heaps)
            state_DQN2 = to_input(heaps)
            is_available = env.check_valid(move_DQN2)
            if not is_available :
                #if the action is not valid, we give the agent a negative reward
                reward = torch.tensor([-1], device=device)
                next_state = None
                if update == True :
                    player_DQN.memory_push(state_DQN2, move_DQN2, next_state, reward)
                    loss = player_DQN.optimize()

                env.end = True
            else : #if the action is valid, we make a step
                heaps, done, _ = env.step(move_DQN2)
                if done : #if the game is finished (done == True), then we give the agent a reward of 1.
                    reward2 = torch.tensor([1], device=device)
                    reward1 = torch.tensor([-1], device=device)
                    next_state = None
                    if update == True:
                        if i > 0 :
                            player_DQN.memory_push(state_DQN1, move_DQN1, state_DQN2, reward1)
                        player_DQN.memory_push(state_DQN2, move_DQN2, next_state, reward2)
                        loss = player_DQN.optimize()
                else:
                  if i > 0 and update == True:
                    # Store the transition in memory
                    reward = torch.tensor([env.reward(player = 1)], device=device)
                    next_state2 = to_input(heaps)
                    player_DQN.memory_push(state_DQN1, move_DQN1, next_state = next_state2, reward = reward)
                    loss = player_DQN.optimize()

        if update == True:
          player_DQN.update_target()

        i += 1

    return reward, loss

def Q11(policy_net : DQN, target_net: DQN, memory : ReplayMemory, nb_games : int = 20000, eps : float = 0.1,
         eps_opt : float = 0.5, step : int = 250, GAMMA : float= 0.99, buffer_size : int = 10000, 
         BATCH_SIZE : int = 64, TARGET_UPDATE : int = 500, seed = None, question : str = 'q3-11',
          nb_samples : int = 5, save : bool = True):
    """
    Implements the solution to the 11th question
    - inputs: 
        - policy_net : the policy network of the DQN player. dtype : DQN
        - target_net : the target network of the DQN player. dtype : DQN
        - memory : memory of the DQN player. dtype : ReplayMemory
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps: espilon associated to the DQN-player (probability of playing at random, i.e. epsilon-greedy) Default: 0.1, dtype: float
        - eps_opt: epsilon associated to the optimal player. Default: 0.5, dtype: float 
        - step: number of games to play before calculating the average reward and average loss. Default: 250, dtype: int
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size of the DQN player. Default: 10000, dtype: float
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q3-11', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - two figures which represent the average reward, respectively loss, every "step" games. 
    """
    Rewards = np.zeros(int(nb_games / step))
    Steps = np.zeros(int(nb_games / step))
    Losses = np.zeros(int(nb_games / step))
    for s in range(nb_samples):
        total_reward = 0.0
        total_loss = 0.0
        env = NimEnv(seed = seed)
        playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
        playerDQN = DQN_Player(player = 1, policy_net = policy_net, target_net= target_net, memory=memory, EPS_GREEDY = eps, GAMMA = GAMMA, buffer_size = buffer_size, BATCH_SIZE = BATCH_SIZE, TARGET_UPDATE = TARGET_UPDATE) 
        for i in range(nb_games):
            if i%step ==0:
                print('New game : ', i)
            # switch turns at every game
            if i % 2 == 0:
                playerOpt.player = 0
                playerDQN.player = 1
            else:
                playerOpt.player = 1
                playerDQN.player = 0
            
            new_reward, new_loss = DQN_one_game(playerDQN, playerOpt, env)
            total_reward += new_reward
            if new_loss != None: #the loss might be None if the opt. player directly wins.
                total_loss += new_loss
            if i % step == step - 1:
                Rewards[i // step] += total_reward / step
                Losses[i // step] += total_loss / step
                Steps[i // step] = i
                total_reward = 0.0
                total_loss = 0.0

            env.reset(seed = seed)
    Rewards = Rewards / nb_samples
    Losses = Losses / nb_samples

    plt.figure(figsize = (9, 8))
    plt.plot(Steps, Rewards)
    plt.title('Evolution of average reward every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for DQN-player')
    if save :
        plt.savefig('./Data/' + question + '_rewards' + str(nb_samples) + 'samples.png')
    plt.show()

    plt.figure(figsize = (9, 8))
    plt.plot(Steps, Losses)
    plt.title('Evolution of average loss every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average loss for DQN-player')
    if save :
        plt.savefig('./Data/' + question + '_losses' + str(nb_samples) + 'samples.png')
        playerDQN.save_net(question)
    plt.show()

class Memory_Q12(object):
    """ Object that can only contain a transition: state, action, next_state and reward.
    For Q12: without the replay buffer and with a batch size of 1. 
    At every step, we will update the network by using only the latest transition."""
    def __init__(self):
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

class DQN_Player_no_RB(DQN_Player):
    def __init__(self, player: int, policy_net : DQN(), target_net : DQN(), memory : Memory_Q12(), 
                EPS_GREEDY: float = 0.2, GAMMA : float = 0.99, BATCH_SIZE: int = 1, TARGET_UPDATE: int = 500):
        """
    Implements a DQN player, without replay buffer (Q.12)
    - inputs: 
        - policy_net : the policy network of the DQN player. dtype : DQN
        - target_net : the target network of the DQN player. dtype : DQN
        - memory : memory of the DQN player. dtype : Memory_Q12
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - EPS_GREEDY: espilon associated to the DQN-player (probability of playing at random, i.e. epsilon-greedy) Default: 0.1, dtype: float
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        """
        super(DQN_Player, self).__init__(player = player)
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.player = player
        self.EPS_GREEDY = EPS_GREEDY
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.buffer_size = BATCH_SIZE
        self.TARGET_UPDATE = TARGET_UPDATE
        self.count = 0 #to count when to update target_net.

        self.optimizer = optim.Adam(policy_net.parameters(), lr = 1e-4)

    def optimize(self):  
        """Optimization step. It is different than DQN_player as the memory changed. """ 
        #the batch consist of the memory (of size 1)     
        batch = self.memory 
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(batch.next_state != None)
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        if batch.next_state != None : #is false if the list is empty
            non_final_next_states = batch.next_state
        else : #if there is no non-final next states
            non_final_next_states = torch.empty(0) 

        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch) #64 x 21
        state_action_values = state_action_values.gather(1, ((action_batch[::2]-1)*7+(action_batch[1::2]-1)).view(self.BATCH_SIZE, 1)) #Batch_size x 1

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        if len(non_final_next_states) > 0 :
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() 
            # max(1) : take the maximum per batch on the 21 possibilities. [0]: take the max and not the argmax
                
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  #64

        # Compute Huber loss
        criterion = nn.HuberLoss()
        loss = criterion(state_action_values.squeeze(), expected_state_action_values.squeeze().detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def memory_push(self, state, action, next_state, reward): 
        """push the transition in the memory."""
        self.memory.push(state, action, next_state, reward)

def Q12(policy_net : DQN(), target_net : DQN(), memory : Memory_Q12(), nb_games : int = 20000, 
        eps : float = 0.1, eps_opt : float = 0.5, step : int = 250, GAMMA : float = 0.99, 
        BATCH_SIZE: int = 1, TARGET_UPDATE: int = 500, seed = None, 
        question : str = 'q3-12', nb_samples : int = 5, save = True):
    """
    Implements the solution to the 12th question
    - inputs: 
        - policy_net : the policy network of the DQN player. dtype : DQN
        - target_net : the target network of the DQN player. dtype : DQN
        - memory : memory of the DQN player. dtype : Memory_Q12
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps: espilon associated to the DQN-player (probability of playing at random, i.e. epsilon-greedy) Default: 0.1, dtype: float
        - eps_opt: epsilon associated to the optimal player. Default: 0.5, dtype: float 
        - step: number of games to play before calculating the average reward and average loss. Default: 250, dtype: int
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q3-12', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - two figures which represent the average reward, respectively loss, every "step" games. 
    """

    Rewards = np.zeros(int(nb_games / step))
    Steps = np.zeros(int(nb_games / step))
    Losses = np.zeros(int(nb_games / step))
    for s in range(nb_samples):
        total_reward = 0.0
        total_loss = 0.0
        env = NimEnv(seed = seed)
        playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
        playerDQN = DQN_Player_no_RB(player = 1, policy_net = policy_net, target_net= target_net, memory=memory,
                                     EPS_GREEDY = eps, GAMMA = GAMMA, BATCH_SIZE = BATCH_SIZE, TARGET_UPDATE = TARGET_UPDATE) 
        for i in range(nb_games):
            if i%step ==0:
                print('New game : ', i)
            # switch turns at every game
            if i % 2 == 0:
                playerOpt.player = 0
                playerDQN.player = 1
            else:
                playerOpt.player = 1
                playerDQN.player = 0
            
            new_reward, new_loss = DQN_one_game(playerDQN, playerOpt, env)
            total_reward += new_reward
            if new_loss != None: #the loss might be None if the opt. player directly wins.
                total_loss += new_loss
            if i % step == step - 1:
                Rewards[i // step] += total_reward / step
                Losses[i // step] += total_loss / step
                Steps[i // step] = i
                total_reward = 0.0
                total_loss = 0.0

            env.reset(seed = seed)
    Rewards = Rewards / nb_samples
    Losses = Losses / nb_samples

    plt.figure(figsize = (9, 8))
    plt.plot(Steps, Rewards)
    plt.title('Evolution of average reward every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for DQN-player')
    if save :
        plt.savefig('./Data/' + question + '_rewards' + str(nb_samples) + 'samples.png')
        playerDQN.save_net(question)
    plt.show()

    plt.figure(figsize = (9, 8))
    plt.plot(Steps, Losses)
    plt.title('Evolution of average loss every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average loss for DQN-player')
    if save :
        plt.savefig('./Data/' + question + '_losses' + str(nb_samples) + 'samples.png')
    plt.show()

def Q13(N_star, nb_games : int = 20000, eps_min : float = 0.1, eps_max : float = 0.8, GAMMA : float = 0.99, 
        buffer_size : int = 10000, BATCH_SIZE : int = 64, TARGET_UPDATE : int = 500,
        step : int = 250, seed = None, question : str = 'q3-13', nb_samples : int = 5, save : bool = True):
    
    """
    Implements the solution to the 13th question
    - inputs: 
        - N_star: a list containing the values of different n*. dtype: list of int
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size of the DQN player. Default: 10 000. dtype : int.
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-4', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each n* as two dictionnaries
    """

    
    fig, axs = plt.subplots(2,1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    for j, n_star in enumerate(N_star):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Rewards = []
        Steps = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            total_reward = 0.0
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star)) 
            playerOpt = OptimalPlayer(epsilon = 0.5, player = 0)
            policy_net = DQN().to(device)
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            memory = ReplayMemory(buffer_size)
            playerDQN = DQN_Player(player = 1, policy_net = policy_net, target_net= target_net, memory=memory,
                                                EPS_GREEDY = eps, GAMMA = GAMMA, buffer_size = buffer_size, BATCH_SIZE = BATCH_SIZE,
                                                TARGET_UPDATE = TARGET_UPDATE) 
            
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerDQN.player = 1
                else:
                    playerOpt.player = 1
                    playerDQN.player = 0

                new_reward, _ = DQN_one_game(playerDQN, playerOpt, env)
                total_reward += new_reward

                if i % step == step - 1:
                    Rewards.append(total_reward / step)
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    # set epsilon to 0 for testing :
                    playerDQN.EPS_GREEDY = 0
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerDQN.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerDQN.player = 0
                            new_reward_mopt, _ = DQN_one_game(playerDQN, new_playerOpt, new_env, update = False)
                            mopt += new_reward_mopt
                            new_env.reset()   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerDQN.player = 1
                            else:
                                playerRand.player = 1
                                playerDQN.player = 0
                            new_reward_mrand, _ = DQN_one_game(playerDQN, playerRand, new_env, update = False)
                            mrand += new_reward_mrand
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                
                env.reset()
                # set the new epsilon for the next
                playerDQN.EPS_GREEDY = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$n^* = {}$".format(n_star))
        Final_Mopt["{}".format(n_star)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(n_star)] = Mrand[-1] / nb_samples
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n*')
    ax2.set_title('Evolution of Mrand for different n*')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
        playerDQN.save_net(question)
    return Final_Mopt, Final_Mrand

def Q14(Eps_opt, n_star = 1000, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, GAMMA = 0.99, buffer_size = 10000, 
        BATCH_SIZE = 64, TARGET_UPDATE = 500, step = 250, seed = None, question = 'q3-14', nb_samples = 5, save = True):
    
    """
    Implements the solution to the 14th question
    - inputs: 
        - Eps_opt: a list containing the values of different eps_opt for the optimal player. dtype: list of floats
        - n_star: the value of n* found in the previous questions to be the best for the DQN-player. Default: 1000, dtype: int
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size of the DQN player. Default: 10 000. dtype : int.
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-4', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each n* as two dictionnaries
    """
    Eps_opt = list(Eps_opt)
    
    fig, axs = plt.subplots(2,1, figsize = (9,13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    for j, eps_opt in enumerate(Eps_opt):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Rewards = []
        Steps = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            total_reward = 0.0
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star)) 
            playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
            policy_net = DQN().to(device)
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            memory = ReplayMemory(buffer_size)
            playerDQN = DQN_Player(player = 1, policy_net = policy_net, target_net= target_net, memory=memory,
                                                EPS_GREEDY = eps, GAMMA = GAMMA, buffer_size = buffer_size, BATCH_SIZE = BATCH_SIZE,
                                                TARGET_UPDATE = TARGET_UPDATE) 
            
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerDQN.player = 1
                else:
                    playerOpt.player = 1
                    playerDQN.player = 0

                new_reward, _ = DQN_one_game(playerDQN, playerOpt, env)
                total_reward += new_reward

                if i % step == step - 1:
                    Rewards.append(total_reward / step)
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    # set epsilon to 0 for testing :
                    playerDQN.EPS_GREEDY = 0
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerDQN.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerDQN.player = 0
                            new_reward_mopt, _ = DQN_one_game(playerDQN, new_playerOpt, new_env, update = False)
                            mopt += new_reward_mopt
                            new_env.reset()   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerDQN.player = 1
                            else:
                                playerRand.player = 1
                                playerDQN.player = 0
                            new_reward_mrand, _ = DQN_one_game(playerDQN, playerRand, new_env, update = False)
                            mrand += new_reward_mrand
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                
                env.reset()
                playerDQN.EPS_GREEDY = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$\varepsilon_o = {}$".format(eps_opt))
        Final_Mopt["{}".format(n_star)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(n_star)] = Mrand[-1] / nb_samples
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title(r'Evolution of Mopt for different $\epsilon_{opt}$')
    ax2.set_title(r'Evolution of Mrand for different $\epsilon_{opt}$')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
        else:
            plt.savefig('./Data/' + question + '.png')
        playerDQN.save_net(question)
    return Final_Mopt, Final_Mrand

def Q16(Eps, nb_games = 20000, GAMMA = 0.99, buffer_size = 10000, BATCH_SIZE = 64, TARGET_UPDATE = 500,
         step = 250, seed = None, question = 'q3-16', nb_samples = 5, save = True):
    """
    Implements the solution to the 16th question. The DQN-player is trained by playing against itself.
    - inputs: 
        - Eps: a list containing the values of different eps for the DQN player. dtype: list of float
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size of the DQN player. Default: 10 000. dtype : int.
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q3-16', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each eps. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand.
    """
    Eps = list(Eps)
    #wf.Q7_warning(Eps, nb_games, alpha, gamma, step, question, nb_samples, save)
    
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    for j, eps in enumerate(Eps):
        print("eps : ", eps)
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            env = NimEnv(seed = seed)

            policy_net = DQN().to(device)
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            memory = ReplayMemory(buffer_size)
            playerDQN = DQN_Player(player = 0, policy_net = policy_net, target_net= target_net, memory=memory,
                                                EPS_GREEDY = eps, GAMMA = GAMMA, buffer_size = buffer_size, BATCH_SIZE = BATCH_SIZE,
                                                TARGET_UPDATE = TARGET_UPDATE)
            
            
            for i in range(nb_games):
                DQN_one_game_vs_self(playerDQN, env)
                
                if i % step == step - 1:
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    # set epsilon to 0 for testing :
                    playerDQN.EPS_GREEDY = 0
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerOpt.player = 0
                                playerDQN.player = 1
                            else:
                                playerOpt.player = 1
                                playerDQN.player = 0
                            new_reward, _ = DQN_one_game(playerDQN, playerOpt, env = new_env, update = False)
                            mopt += new_reward
                            new_env.reset()   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerDQN.player = 1
                            else:
                                playerRand.player = 1
                                playerDQN.player = 0
                            new_reward, _ = DQN_one_game(playerDQN, playerRand, env = new_env, update = False)
                            mrand += new_reward
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                    # set back espilon (no longer testing)
                    playerDQN.EPS_GREEDY = eps
                
                env.reset(seed = seed)
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$\varepsilon = {}$".format(eps))
        Final_Mopt["{}".format(eps)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(eps)] = Mrand[-1] / nb_samples
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different epsilon')
    ax2.set_title('Evolution of Mrand for different epsilon')
    ax.set_xlabel('Number of games played against itself')
    ax2.set_xlabel('Number of games played against itself')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
            #plt.savefig("/content/drive/MyDrive/ColabNotebooks/ANN/Data/" + question + '_' + str(nb_samples) + "_samples.png")
        else:
            plt.savefig('./Data/' + question + '.png')
            #plt.savefig('/content/drive/MyDrive/ColabNotebooks/ANN/Data/'+ question + '.png')
        playerDQN.save_net(question)
    return Final_Mopt, Final_Mrand

def Q17(N_star, nb_games : int = 20000, eps_min : float = 0.1, eps_max : float = 0.8, GAMMA : float = 0.99, 
        buffer_size : int = 10000, BATCH_SIZE : int = 64, TARGET_UPDATE : int = 500,
        step : int = 250, seed = None, question : str = 'q3-17', nb_samples : int = 5, save : bool = True):
    
    """
    Implements the solution to the 17th question
    - inputs: 
        - N_star: a list containing the values of different n*. dtype: list of int
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size of the DQN player. Default: 10 000. dtype : int.
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - step: number of games to play before calculating the average reward. Default: 250, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q2-4', dtype: str
        - nb_samples: if this number is higher than 1, the 'nb_games' are played several times and then averaged in order to take into account the schocasticity of te problem. Default: 5, dtype: int
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - outputs: 
        - a figure with two subplots representing respectively the performance against the optimal player (Mopt) and against a totally random player (Mrand) with a plot for each n*. The performance is averaged on 500 games played 5 times to take stochasticity into account. 
        According the the value of the argument 'nb_samples', two different figures can be produced. Figures are saved in a folder Data if the argument 'save' is set to True.
        - returns the final Mopt, Mrand for each n* as two dictionnaries
    """
    N_star = list(N_star)
    
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    Final_Mopt = {}
    Final_Mrand = {}
    for j, n_star in enumerate(N_star):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star)) 
            policy_net = DQN().to(device)
            target_net = DQN().to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            memory = ReplayMemory(buffer_size)
            playerDQN = DQN_Player(player = 1, policy_net = policy_net, target_net= target_net, memory=memory,
                                                EPS_GREEDY = eps, GAMMA = GAMMA, buffer_size = buffer_size, BATCH_SIZE = BATCH_SIZE,
                                                TARGET_UPDATE = TARGET_UPDATE) 
            
            for i in range(nb_games):
                DQN_one_game_vs_self(playerDQN, env)
    
                if i % step == step - 1:
                    Steps[i // step] = i
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    playerDQN.EPS_GREEDY = 0
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerDQN.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerDQN.player = 0
                            new_reward_mopt, _ = DQN_one_game(playerDQN, new_playerOpt, new_env, update = False)
                            mopt += new_reward_mopt
                            new_env.reset()   
                
                        # compute M_rand
                        playerRand = OptimalPlayer(epsilon = 1, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerRand.player = 0
                                playerDQN.player = 1
                            else:
                                playerRand.player = 1
                                playerDQN.player = 0
                            new_reward_mrand, _ = DQN_one_game(playerDQN, playerRand, new_env, update = False)
                            mrand += new_reward_mrand
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                
                env.reset()
                playerDQN.EPS_GREEDY = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$n^* = {}$".format(n_star))
        Final_Mopt["{}".format(n_star)] = Mopt[-1] / nb_samples
        Final_Mrand["{}".format(n_star)] = Mrand[-1] / nb_samples
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n*')
    ax2.set_title('Evolution of Mrand for different n*')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if save:
        if nb_samples > 1:
            plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
            #plt.savefig("/content/drive/MyDrive/ColabNotebooks/ANN/Data/" + question + '_' + str(nb_samples) + "_samples.png")
        else:
            plt.savefig('./Data/' + question + '.png')
            #plt.savefig('/content/drive/MyDrive/ColabNotebooks/ANN/Data/'+ question + '.png')
        playerDQN.save_net(question)

    return Final_Mopt, Final_Mrand

def Q19_train(n_star = 5000, nb_games : int = 20000, eps_min : float = 0.1, eps_max : float = 0.8, GAMMA : float = 0.99, 
        buffer_size : int = 10000, BATCH_SIZE : int = 64, TARGET_UPDATE : int = 500,
        seed = None, question : str = 'q3-19_train', save : bool = True):
    
    """
    Implements the training of the best self-learning DQN player and save the model of its policy network.
    - inputs: 
        - n_star: the value n*. Default: 5000, dtype: int
        - nb_games: the number of games to play. Default: 20000, dtype: int
        - eps_min: the minimal value for the exploration level of the QL-player. Default: 0.1, dtype: float
        - eps_max: the maximal value for the exploration level of the QL-player. Default: 0.8, dtype: float
        - GAMMA : discount factor of the DQN player. Default: 0.99, dtype: float
        - buffer_size : buffer size of the DQN player. Default: 10 000. dtype : int.
        - BATCH_SIZE : batch size of the DQN player. Default: 1, dtype: int
        - TARGET_UPDATE : number of games after the target network of the DQN player is updated. Default: 500, dtype: int
        - seed: the user can set a given seed for reproducibility. Default: None
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q3-19_train', dtype: str
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool

    """

    env = NimEnv(seed = seed)
    eps = max(eps_min, eps_max * (1 - 1 / n_star)) 
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    memory = ReplayMemory(buffer_size)
    playerDQN = DQN_Player(player = 1, policy_net = policy_net, target_net= target_net, memory=memory,
                                        EPS_GREEDY = eps, GAMMA = GAMMA, buffer_size = buffer_size, BATCH_SIZE = BATCH_SIZE,
                                        TARGET_UPDATE = TARGET_UPDATE) 
    
    #training against himself for nb_games games.
    for i in range(nb_games):
        DQN_one_game_vs_self(playerDQN, env)         
        env.reset()
        playerDQN.EPS_GREEDY = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))

    #save the model
    if save == True:
        PATH = './Data/model' + question + '.pth'
        torch.save(policy_net.state_dict(), PATH)

def Q19(playerDQN : DQN_Player, configs = np.array([[3, 0, 0], [1, 2, 0], [0, 3, 2]]), question = 'q3-19', save = True):
    """
    Implements the solution of the 19th question. 
    inputs: 
        - playerDQN : the agent who predicts the q-values. dtype : DQN_Player
        - configs : array of dimension 3 x 3, representing 3 heaps. The q-values will be predicted from these heaps. 
                    Default: np.array([[3, 0, 0], [1, 2, 0], [0, 3, 2]])
        - question: string used to differentiate between the plots for each question. 
            Only used if 'save' is True. Default: 'q3-19', dtype: str
        - save: if set to False, the plots are only displayed but not saved. Default: True, dtype: bool
    - output: 
        - a figure with 3 subplots representing the q-values predicted by the DQN-player, for each heap.
    """
    first_heap = configs[0, :]
    second_heap= configs[1, :]
    third_heap = configs[2, :]
    qvals1 = playerDQN.predict(first_heap).detach().numpy()
    qvals2 = playerDQN.predict(second_heap).detach().numpy()
    qvals3 = playerDQN.predict(third_heap).detach().numpy()

        
    fig, axs = plt.subplots(3, 1, figsize = (30, 10))
    fig.subplots_adjust(hspace=0.5)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    x_label_list = np.arange(1, 8, 1)
    y_label_list = [1, 2, 3]


    ax1.set_yticks(np.arange(0, 3, 1))
    ax1.set_yticklabels(y_label_list)
    ax1.set_xticks(np.arange(0, 7, 1))
    ax1.set_xticklabels(x_label_list)
    ax1.set_xlabel('Number of sticks')
    ax1.set_ylabel('Heap')
    divider1 = make_axes_locatable(ax1)
    ax1.set_title('Current configuration: ' + str(first_heap[0]) + ' | ' + str(first_heap[1]) + ' | ' + str(first_heap[2]))
    im1 = ax1.imshow(qvals1)
    #color bar on the right
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax = cax1, label = 'Q-values')
    
    
    ax2.set_yticks(np.arange(0, 3, 1))
    ax2.set_yticklabels(y_label_list)
    ax2.set_xticks(np.arange(0, 7, 1))
    ax2.set_xticklabels(x_label_list)
    ax2.set_xlabel('Number of sticks')
    ax2.set_ylabel('Heap')
    divider2 = make_axes_locatable(ax2)
    ax2.set_title('Current configuration: ' + str(second_heap[0]) + ' | ' + str(second_heap[1]) + ' | ' + str(second_heap[2]))
    im2 = ax2.imshow(qvals2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax = cax2, label = 'Q-values')

    ax3.set_yticks(np.arange(0, 3, 1))
    ax3.set_yticklabels(y_label_list)
    ax3.set_xticks(np.arange(0, 7, 1))
    ax3.set_xticklabels(x_label_list)
    ax3.set_xlabel('Number of sticks')
    ax3.set_ylabel('Heap')
    divider3 = make_axes_locatable(ax3)
    ax3.set_title('Current configuration: ' + str(third_heap[0]) + ' | ' + str(third_heap[1]) + ' | ' + str(third_heap[2]))
    im3 = ax3.imshow(qvals3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05) 
    plt.colorbar(im3, cax = cax3, label = 'Q-values')

    if save:
        fig.savefig('./Data/' + question + '.png')

    