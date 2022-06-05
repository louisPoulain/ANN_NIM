# In this file we implement some helpers functions and algorithms dedicated to our tasks

from operator import index
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from nim_env import NimEnv, OptimalPlayer, QL_Player
import WarningFunctions as wf
import warnings
import time

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)
plt.rc('axes',titlesize=20, labelsize = 15)
plt.rc('legend',fontsize=15)
plt.rc('figure',titlesize=20)

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
        - returns the list of rewards
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
        - returns the final Mopt, Mrand for each n* as two dictionnaries
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
        - returns the final Mopt, Mrand for each n* as two dictionnaries
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
        - returns the final Mopt, Mrand for each n* as two dictionnaries and the set of Q-values of the QL-player after all games are played
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

    #v_min = -8
    #v_max = 60

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
    #im1.set_clim(v_min, v_max)
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
    #im2.set_clim(v_min, v_max)
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
    #im3.set_clim(v_min, v_max)
    plt.colorbar(im3, cax = cax3, label = 'Q-values')

    if save:
        fig.savefig('./Data/' + question + '.png')




    '''
    first_config = configs[0]
    second_config = configs[1]
    third_config = configs[2]
    fig, axs = plt.subplots(1, 3, figsize = (18, 6))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    len1 = len(qval[first_config])
    len2 = len(qval[second_config])
    len3 = len(qval[third_config])
    tick_positions1 = np.linspace(1. / len1 , 1, len1, endpoint = False)
    tick_positions2 = np.linspace(1. / len2 , 1, len2, endpoint = False)
    tick_positions3 = np.linspace(1. / len3 , 1, len3, endpoint = False)
    
    keys1 = [t for t in qval[first_config].keys()]
    tick_labels1 = [str(int(first_config[0]) - int(keys1[i][0])) + str(int(first_config[1]) - int(keys1[i][1])) + str(int(first_config[2]) - int(keys1[i][2])) for i in range(len(keys1))]
    qvals1 = [q for q in qval[first_config].values()]
    ax1.set_xticks(tick_positions1)
    ax1.set_xticklabels(tick_labels1)
    ax1.set_xlabel('Possible actions')
    ax1.set_ylabel('Q-values')
    ax1.set_title('Current configuration: ' + str(first_config[0]) + ' | ' + str(first_config[1]) + ' | ' + str(first_config[2]))
    ax1.bar(tick_positions1, qvals1, width = 1. / (2 * len1))
    
    keys2 = [t for t in qval[second_config].keys()]
    tick_labels2 = [str(int(second_config[0]) - int(keys2[i][0])) + str(int(second_config[1]) - int(keys2[i][1])) + str(int(second_config[2]) - int(keys2[i][2])) for i in range(len(keys2))]
    qvals2 = [q for q in qval[second_config].values()]
    ax2.set_xticks(tick_positions2)
    ax2.set_xticklabels(tick_labels2)
    ax2.set_xlabel('Possible actions')
    ax2.set_ylabel('Q-values')
    ax2.set_title('Current configuration: ' + str(second_config[0]) + ' | ' + str(second_config[1]) + ' | ' + str(second_config[2]))
    ax2.bar(tick_positions2, qvals2, width = 1. / (2 * len2))
    
    keys3 = [t for t in qval[third_config].keys()]
    tick_labels3 = [str(int(third_config[0]) - int(keys3[i][0])) + str(int(third_config[1]) - int(keys3[i][1])) + str(int(third_config[2]) - int(keys3[i][2])) for i in range(len(keys3))]
    qvals3 = [q for q in qval[third_config].values()]
    ax3.set_xticks(tick_positions3)
    ax3.set_xticklabels(tick_labels3)
    ax3.set_xlabel('Possible actions')
    ax3.set_ylabel('Q-values')
    ax3.set_title('Current configuration: ' + str(third_config[0]) + ' | ' + str(third_config[1]) + ' | ' + str(third_config[2]))
    ax3.bar(tick_positions3, qvals3, width = 1. / (2 * len3))

    if save:
        fig.savefig('./Data/' + question + '.png')
    '''
    
    
    
    

