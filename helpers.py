# In this file we implement some helpers functions and algorithms dedicated to our tasks

import numpy as np
import matplotlib.pyplot as plt
from nim_env import NimEnv, OptimalPlayer, QL_Player

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=11)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=11)
plt.rc('figure',titlesize=20)

def QL_one_game(playerQL, playerOpt, eps, eps_opt, alpha, gamma, env, update = True):
    heaps, _, _ = env.observe()
    i = 0
    while not env.end:
        if env.current_player == playerOpt.player:
            move = playerOpt.act(heaps)
            heaps, end, winner = env.step(move)
            if i > 0 and update == True:
                playerQL.update_qval(ql_action = action, other_move = move, heaps_before = heaps_before, heaps_after = heaps,
                                     env = env, alpha = alpha, gamma = gamma)
        else:
            
            
            heaps_before = heaps.copy()    
            move, action = playerQL.act(heaps)
            heaps, end, winner = env.step(move)
        
        i += 1
    
    return env.reward(playerQL.player)
                
def Q1(nb_games = 20000, eps = 0.2, eps_opt = 0.5, alpha = 0.1, gamma = 0.99, step = 250, seed = None, question = 'q2-1'):
    Rewards = []
    Steps = []
    total_reward = 0.0
    env = NimEnv(seed = seed)
    playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
    playerQL = QL_Player(epsilon = eps, player = 1)
    #print(playerQL.qvals['746'])
    for i in range(nb_games):
        #print('New game\n')
        # switch turns at every game
        if i % 2 == 0:
            playerOpt.player = 0
            playerQL.player = 1
        else:
            playerOpt.player = 1
            playerQL.player = 0
        
        total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, alpha = alpha, gamma = gamma, env = env)
        if i % step == step - 1:
            Rewards.append(total_reward / step)
            Steps.append(i)
            total_reward = 0.0
        env.reset()
        #print(playerQL.qvals['746'])
    plt.figure(figsize = (7, 7))
    plt.plot(Steps, Rewards)
    plt.title('Evolution of average reward every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for QL-player')
    plt.savefig('./Data/' + question + '.png')
    

def Q2(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, step = 250, seed = None, question = 'q2-2'):
    plt.figure(figsize = (9, 8))
    legend = []
    for j, n_star in enumerate(N_star):
        Rewards = []
        Steps = []
        total_reward = 0.0
        env = NimEnv(seed = seed)
        eps = eps_min # always true for the first game
        playerOpt = OptimalPlayer(epsilon = 0.5, player = 0)
        playerQL = QL_Player(epsilon = eps, player = 1)
        for i in range(nb_games):
            #print('New game\n')
            # switch turns at every game
            if i % 2 == 0:
                playerOpt.player = 0
                playerQL.player = 1
            else:
                playerOpt.player = 1
                playerQL.player = 0
        
            total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, alpha = alpha, gamma = gamma, env = env)
            if i % step == step - 1:
                Rewards.append(total_reward / step)
                Steps.append(i)
                total_reward = 0.0
            env.reset()
            playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        plt.plot(Steps, Rewards)
        legend.append(r"$n_* = {}$".format(n_star))
    plt.legend(legend)
    plt.title('Evolution of average reward with decrease of exploration level')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for QL-player')
    plt.savefig('./Data/' + question + '.png')
    
def Q3(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, step = 250, seed = None, question = 'q2-3'):
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    for j, n_star in enumerate(N_star):
        Mopt = []
        Mrand = []
        Rewards = []
        Steps = []
        total_reward = 0.0
        env = NimEnv(seed = seed)
        eps = eps_min # always true for the first game
        playerOpt = OptimalPlayer(epsilon = 0.5, player = 0)
        playerQL = QL_Player(epsilon = eps, player = 1)
        for i in range(nb_games):
            #print('New game\n')
            # switch turns at every game
            if i % 2 == 0:
                playerOpt.player = 0
                playerQL.player = 1
            else:
                playerOpt.player = 1
                playerQL.player = 0
        
            total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, alpha = alpha, gamma = gamma, env = env)
            if i % step == step - 1:
                Rewards.append(total_reward / step)
                Steps.append(i)
                total_reward = 0.0
                mopt = 0
                mrand = 0
                new_env = NimEnv()
                # compute M_opt
                new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                for k in range(500):
                    if k % 2 == 0:
                        new_playerOpt.player = 0
                        playerQL.player = 1
                    else:
                        new_playerOpt.player = 1
                        playerQL.player = 0
                    mopt += QL_one_game(playerQL, new_playerOpt, eps = playerQL.epsilon, eps_opt = new_playerOpt.epsilon, alpha = alpha, 
                                        gamma = gamma, env = new_env, update = False)
                    new_env.reset()
                    # playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) ???????????????
                Mopt.append(mopt / 500)   
                
                # compute M_rand
                playerRand = OptimalPlayer(epsilon = 1, player = 0)
                for k in range(500):
                    if k % 2 == 0:
                        playerRand.player = 0
                        playerQL.player = 1
                    else:
                        playerRand.player = 1
                        playerQL.player = 0
                    mrand += QL_one_game(playerQL, playerRand, eps = playerQL.epsilon, eps_opt = playerRand.epsilon, alpha = alpha, 
                                        gamma = gamma, env = new_env, update = False)
                    new_env.reset()
                Mrand.append(mrand / 500)
                
            env.reset()
            playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt)
        ax2.plot(Steps, Mrand)
        legend.append(r"$n_* = {}$".format(n_star))
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n')
    ax2.set_title('Evolution of Mrand for different n')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    fig.savefig('./Data/' + question + '.png')
            

    
    
    
    
    

