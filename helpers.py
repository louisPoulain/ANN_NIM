# In this file we implement some helpers functions and algorithms dedicated to our tasks

import numpy as np
from nim_env import NimEnv, OptimalPlayer

def QL_algo(eps = 0.2, eps_opt = 0.5, alpha = 0.1, gamma = 0.99, nb_games = 20000):
    # env
    env = NimEnv(seed = 3)
    
    # players: Q-learning vs opt
    player_opt = OptimalPlayer(epsilon = eps_opt, player = 0)
    player_ql = OptimalPlayer(epsilon = eps, player = 1)
    
    # Allocate Q-values as dictionnary of dictionnaries and initialize them
    Qvals = {}
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                next_actions = {}
                for l in range(i):
                    next_actions[str(l) + str(j) + str(k)] = 0
                for l in range(j):
                    next_actions[str(i) + str(l) + str(k)] = 0
                for l in range(k):
                    next_actions[str(i) + str(j) + str(l)] = 0
                    
                string = str(i) + str(j) + str(k)
                Qvals[string] = next_actions
    
    for i in range(nb_games):
        heaps, _, _ = env.observe()
        s = heaps
        while not env.end:
            if env.current_player == player_opt.player:
                move = player_opt.act(heaps)
                heaps, end, winner = env.step(move)
                s_new = heaps
            else: # il faudra plutôt créer une autre fonction de move pour ça (dans la classe optPlayer ?)
                # choose A from state s
                current_config = str(s[0]) + str(s[1]) + str(s[2])
                i, j, k = max(Qvals[current_config])
                action = [int(i), int(j), int(k)]
                nb_stick, pile_to_take = max(heaps - action), np.argmax(heaps - action)
                move = [pile_to_take, nb_stick]
                
                heaps, end, winner = env.step(move)
                s_new = heaps
                reward = env.reward(env.current_player)
                next_config = str(s_new[0]) + str(s_new[1]) + str(s_new[2])
                
                Q_s_new_a = Qvals[next_config][max(Qvals[next_config])] # in the algo: max Q(S', a) over all a (on peut lui trouver un meilleur nom)
                
                Qvals[current_config][i + j + k] = (1 - alpha) * Qvals[current_config][i + j + k] + alpha * reward + gamma * alpha * Q_s_new_a
            
            s = s_new
            if end:
                env.reset()
                break
                
                
            
            

    
    
    
    
    

