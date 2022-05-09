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

def QL_one_game_vs_self(playerQL, eps, alpha, gamma, env, update = True):
    playerQL1, playerQL2 = playerQL.copy(), playerQL.copy()
    playerQL1.player = 1
    playerQL2.player = 0
    heaps, _, _ = env.observe()
    i = 0
    heaps_after = [[], []]
    Actions = []
    #heaps_before.append(heaps.copy())
    while not env.end:
        if env.current_player == playerQL1.player:
            heaps_before1 = heaps.copy()
            move1, action1 = playerQL1.act(heaps)
            heaps, end, winner = env.step(move1)
            heaps_after[1] = heaps.copy()
            
        else:
            heaps_before2 = heaps.copy()    
            move2, action2 = playerQL.act(heaps)
            heaps, end, winner = env.step(move2)
            heaps_after[0] = heaps.copy()
            
        if i > 0 and update == True:
            if env.current_player == 0: # player 1 just played
                playerQL.player = 0 
                playerQL.update_qval(ql_action = action1, other_move = move2, heaps_before = heaps_before1, 
                                     heaps_after = heaps_after[0],env = env, alpha = alpha, gamma = gamma)
                playerQL.player = 1
                playerQL.update_qval(ql_action = action2, other_move = move1, heaps_before = heaps_before2, heaps_after = heaps_after[1],
                                     env = env, alpha = alpha, gamma = gamma)
            else:
                playerQL.player = 1
                playerQL.update_qval(ql_action = action2, other_move = move1, heaps_before = heaps_before2, heaps_after = heaps_after[1],
                                     env = env, alpha = alpha, gamma = gamma)
                playerQL.player = 0 
                playerQL.update_qval(ql_action = action1, other_move = move2, heaps_before = heaps_before1, 
                                     heaps_after = heaps_after[0],env = env, alpha = alpha, gamma = gamma)
            
                
        playerQL1.qvals = playerQL.qvals.copy()
        playerQL2.qvals = playerQL.qvals.copy()
        """
        playerQL.player = env.current_player
        move, action = playerQL.act(heaps)
        Actions.append(action.copy())
        heaps, end, winner = env.step(move)
        heaps_before.append(heaps.copy())
        if i > 0 and update == True:
            playerQL.update_qval(ql_action = Actions[0], other_move = move, heaps_before = heaps_before[0], 
                                 heaps_after = heaps, env = env, alpha = alpha, gamma = gamma)
            heaps_before[0] = heaps_before[-2]
            Actions[0] = Actions[-1]
        """
        i += 1
                
def Q1(nb_games = 20000, eps = 0.2, eps_opt = 0.5, alpha = 0.1, gamma = 0.99, step = 250, seed = None, question = 'q2-1', nb_samples = 5):
    assert isinstance(nb_samples, int), "The number of samples has to be integer"
    assert nb_samples > 0, "The number of samples cannot be less than 1"
    plt.figure(figsize = (9, 8))
    Rewards = np.zeros(int(nb_games / step))
    Steps = np.zeros(int(nb_games / step))
    for s in range(nb_samples):
        total_reward = 0.0
        env = NimEnv(seed = seed)
        playerOpt = OptimalPlayer(epsilon = eps_opt, player = 0)
        playerQL = QL_Player(epsilon = eps, player = 1)
        for i in range(nb_games):
            # switch turns at every game
            if i % 2 == 0:
                playerOpt.player = 0
                playerQL.player = 1
            else:
                playerOpt.player = 1
                playerQL.player = 0
        
            total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, 
                                        alpha = alpha, gamma = gamma, env = env)
            if i % step == step - 1:
                Rewards[i // step] += total_reward / step
                Steps[i // step] = i
                total_reward = 0.0
            env.reset()
    Rewards = Rewards / nb_samples
    plt.plot(Steps, Rewards)
    plt.title('Evolution of average reward every 250 games')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for QL-player (' + str(eps) + ')')
    if nb_samples > 1:
        plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
    else:
        plt.savefig('./Data/' + question + '.png')
    

def Q2(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, 
       step = 250, seed = None, question = 'q2-2', nb_samples = 5):
    assert isinstance(nb_samples, int), "The number of samples has to be integer"
    assert nb_samples > 0, "The number of samples cannot be less than 1"
    plt.figure(figsize = (9, 8))
    legend = []
    for j, n_star in enumerate(N_star):
        Rewards = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        for s in range(nb_samples):
            env = NimEnv(seed = seed)
            eps = max(eps_min, eps_max * (1 - 1 / n_star))
            playerOpt = OptimalPlayer(epsilon = 0.05, player = 0)
            playerQL = QL_Player(epsilon = eps, player = 1)
            total_reward = 0.0
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerQL.player = 1
                else:
                    playerOpt.player = 1
                    playerQL.player = 0
        
                total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, 
                                            alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Rewards[i // step] += total_reward / step
                    total_reward = 0.
                    Steps[i // step] = i
                env.reset()
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        Rewards = Rewards / nb_samples
        plt.plot(Steps, Rewards)
        legend.append(r"$n_* = {}$".format(n_star))
    plt.legend(legend)
    plt.title('Evolution of average reward with decrease of exploration level')
    plt.xlabel('Number of games played')
    plt.ylabel('Average reward for QL-player')
    if nb_samples > 1:
        plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
    else:
        plt.savefig('./Data/' + question + '.png')
    
def Q3(N_star, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, 
       step = 250, seed = None, question = 'q2-3', nb_samples = 5):
    assert isinstance(nb_samples, int), "The number of samples has to be integer"
    assert nb_samples > 0, "The number of samples cannot be less than 1"
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
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
            playerQL = QL_Player(epsilon = eps, player = 1)
            
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerQL.player = 1
                else:
                    playerOpt.player = 1
                    playerQL.player = 0
        
                total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, 
                                        alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Rewards.append(total_reward / step)
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, new_playerOpt, eps = playerQL.epsilon, eps_opt = new_playerOpt.epsilon, 
                                                alpha = alpha, gamma = gamma, env = new_env, update = False)
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
                            mrand += QL_one_game(playerQL, playerRand, eps = playerQL.epsilon, eps_opt = playerRand.epsilon, 
                                                 alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                
                env.reset()
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$n_* = {}$".format(n_star))
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n*')
    ax2.set_title('Evolution of Mrand for different n*')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if nb_samples > 1:
        plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
    else:
        plt.savefig('./Data/' + question + '.png')
        
def Q4(Eps_opt, n_star = 1000, nb_games = 20000, eps_min = 0.1, eps_max = 0.8, alpha = 0.1, gamma = 0.99, 
       step = 250, seed = None, question = 'q2-4', nb_samples = 5):
    assert isinstance(nb_samples, int), "The number of samples has to be integer"
    assert nb_samples > 0, "The number of samples cannot be less than 1"
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
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
            playerQL = QL_Player(epsilon = eps, player = 1)
            
            for i in range(nb_games):
                # switch turns at every game
                if i % 2 == 0:
                    playerOpt.player = 0
                    playerQL.player = 1
                else:
                    playerOpt.player = 1
                    playerQL.player = 0
        
                total_reward += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, 
                                        alpha = alpha, gamma = gamma, env = env)
                if i % step == step - 1:
                    Rewards.append(total_reward / step)
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        new_playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                new_playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                new_playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, new_playerOpt, eps = playerQL.epsilon, eps_opt = new_playerOpt.epsilon, 
                                                alpha = alpha, gamma = gamma, env = new_env, update = False)
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
                            mrand += QL_one_game(playerQL, playerRand, eps = playerQL.epsilon, eps_opt = playerRand.epsilon, 
                                                 alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                
                env.reset()
                playerQL.epsilon = max(eps_min, eps_max * (1 - (i + 2) / n_star)) # change eps for the next game (current game is (i+1))
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$\varepsilon_o = {}$".format(eps_opt))
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different n*')
    ax2.set_title('Evolution of Mrand for different n*')
    ax.set_xlabel('Number of games played')
    ax2.set_xlabel('Number of games played')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if nb_samples > 1:
        plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
    else:
        plt.savefig('./Data/' + question + '.png')
        
        
def Q7(Eps, nb_games = 20000, alpha = 0.1, gamma = 0.99, step = 250, seed = None, question = 'q2-7', nb_samples = 5):
    assert isinstance(nb_samples, int), "The number of samples has to be integer"
    assert nb_samples > 0, "The number of samples cannot be less than 1"
    fig, axs = plt.subplots(2, 1, figsize = (9, 13))
    ax = axs[0]
    ax2 = axs[1]
    legend = []
    for j, eps in enumerate(Eps):
        Mopt = np.zeros(int(nb_games / step))
        Mrand = np.zeros(int(nb_games / step))
        Steps = np.zeros(int(nb_games / step))
        for l in range(nb_samples):
            env = NimEnv(seed = seed)
            playerQL = QL_Player(epsilon = eps, player = 0)
            
            for i in range(nb_games):
                print('new games\n')
                QL_one_game_vs_self(playerQL, eps = playerQL.epsilon, 
                                        alpha = alpha, gamma = gamma, env = env)
                #print(playerQL.qvals['210'], playerQL.qvals['250'])
                if i % step == step - 1:
                    Steps[i // step] = i
                    total_reward = 0.0
                    mopt = 0
                    mrand = 0
                    new_env = NimEnv()
                    for m in range(5):  # here we run for several different seeds
                        # compute M_opt
                        playerOpt = OptimalPlayer(epsilon = 0, player = 0)
                        for k in range(500):
                            if k % 2 == 0:
                                playerOpt.player = 0
                                playerQL.player = 1
                            else:
                                playerOpt.player = 1
                                playerQL.player = 0
                            mopt += QL_one_game(playerQL, playerOpt, eps = playerQL.epsilon, eps_opt = playerOpt.epsilon, 
                                                alpha = alpha, gamma = gamma, env = new_env, update = False)
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
                            mrand += QL_one_game(playerQL, playerRand, eps = playerQL.epsilon, eps_opt = playerRand.epsilon, 
                                                 alpha = alpha, gamma = gamma, env = new_env, update = False)
                            new_env.reset()
                    Mrand[i // step] += mrand / (500 * 5)
                    Mopt[i // step] += mopt / (500 * 5)
                
                env.reset(seed = seed)
        
        ax.plot(Steps, Mopt / nb_samples)
        ax2.plot(Steps, Mrand / nb_samples)
        legend.append(r"$\varepsilon = {}$".format(eps))
    
    ax.legend(legend)
    ax2.legend(legend)
    ax.set_title('Evolution of Mopt for different epislon')
    ax2.set_title('Evolution of Mrand for different epsilon')
    ax.set_xlabel('Number of games played against itself')
    ax2.set_xlabel('Number of games played against itself')
    ax.set_ylabel(r'$M_{opt}$')
    ax2.set_ylabel(r'$M_{rand}$')
    if nb_samples > 1:
        plt.savefig('./Data/' + question + '_' + str(nb_samples) + '_samples.png')
    else:
        plt.savefig('./Data/' + question + '.png')
            

    
    
    
    
    

