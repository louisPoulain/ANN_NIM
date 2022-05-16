import random
import numpy as np

class NimEnv:
    def __init__(self, seed=None):
        self.n_heap = 3
        self.n_agents = 2
        self.current_player = 0
        self.winner = None
        self.end = False
        self.num_step = 0

        if seed is not None:
            random.seed(seed)
        self.heaps = random.sample(range(1, 8), 3)
        self.heap_avail = [True, True, True]
        self.heap_keys = ["1", "2", "3"]
        self.winner = None

    def check_valid(self, action):
        h, n = map(int, action)
        if not self.heap_avail[h - 1]:
            return False
        if n < 1:
            return False
        if n > self.heaps[h - 1]:
            return False
        return True

    def step(self, action):
        """
        step method takin an action as input

        Parameters
        ----------
        action : list(int)
            action[0] = 1, 2, 3 is the selected heap to take from
            action[1] is the number of objects to take from the heap

        Returns
        -------
        getObservation()
            State space (printable).
        reward : tuple
            (0,0) when not in final state, +1 for winner and -1 for loser
            otherwise.
        done : bool
            is the game finished.
        dict
            dunno.

        """

        # extracting integer values h: heap id, n: nb objects to take
        h, n = map(int, action)

        assert self.heap_avail[h - 1], "The selected heap is already empty"
        assert n >= 1, "You must take at least 1 object from the heap"
        assert (n <= self.heaps[h - 1]), "You cannot take more objects than there are in the heap"

        self.heaps[h - 1] -= n  # core of the action

        if self.heaps[h - 1] == 0:
            self.heap_avail[h - 1] = False

        reward = (0, 0)
        done = False
        if self.heap_avail.count(True) == 0:
            done = True
            self.winner = self.current_player

        self.end = done
        # update
        self.num_step += 1
        self.current_player = 0 if self.num_step % 2 == 0 else 1
        next_heaps = self.heaps[:]
        return self.heaps, self.end, self.winner

    def observe(self):
        return self.heaps, self.end, self.winner

    def reward(self, player=0):
        if self.end:
            if self.winner is None:
                return 0
            else:
                return 1 if player == self.winner else -1
        else:
            return 0

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.heaps = random.sample(range(1, 8), 3)
        self.heap_avail = [True, True, True]
        self.current_player = 0
        self.winner = None
        self.end = False
        self.num_step = 0
        return self.heaps.copy()

    def render(self, simple=False):
        if simple:
            print(self.heaps)
        else:
            print(u"\u2500" * 35)
            for i in range(len(self.heaps)):
                print(
                    "Heap {}: {:15s} \t ({})".format(
                        self.heap_keys[i], "|" * self.heaps[i], self.heaps[i]
                    )
                )
                print(u"\u2500" * 35)


class OptimalPlayer:
    '''
    Description:
        A class to implement an epsilon-greedy optimal player in Nim.

    About optimial policy:
        Optimal policy relying on nim sum (binary XOR) taken from
        https://en.wikipedia.org/wiki/Nim#Example_implementation
        We play normal (i.e. not misere) game: the player taking the last object wins

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the optimal action
            at any given time.

    '''
    def __init__(self, epsilon=0.2, player=0):
        self.epsilon = epsilon
        self.player = player #  0 or 1

    def set_player(self, player = 0, j=-1):
        self.player = player
        if j != -1:
            self.player = 0 if j % 2 == 0 else 1

    def randomMove(self, heaps):
        """
        Random policy (then optimal when obvious):
            - Select an available heap
            - Select a random integer between 1 and the number of objects in this heap.
            
        Parameters
        ----------
        heaps : list of integers
                list of heap sizes.

        Returns
        -------
        move : list
            move[0] is the heap to take from (starts at 1)
            move[1] is the number of obj to take from heap #move[0]
        """
        # the indexes of the heaps available are given by
        heaps_avail = [i for i in range(len(heaps)) if heaps[i] > 0]
        chosen_heap = random.choice(heaps_avail)
        n_obj = random.choice(range(1, heaps[chosen_heap] + 1))
        move = [chosen_heap + 1, n_obj]

        return move
    
    def compute_nim_sum(self, heaps):
        """
        The nim sum is defined as the bitwise XOR operation,
        this is implemented in python with the native caret (^) operator.

        The bitwise XOR operation is such that:
            if we have heaps = [3, 4, 5],
            it can be written in bits as heaps = [011, 100, 101],
            and the bitwise XOR problem gives 010 = 2 (the nim sum is 2)

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.

        Returns
        -------
        nim : int
            nim sum of all heap sizes.

        """
        nim = 0
        for i in heaps:
            nim = nim ^ i # ^ = XOR operation in 
        return nim

    def act(self, heaps, **kwargs):
        """
        Optimal policy relying on nim sum (binary XOR) taken from
        https://en.wikipedia.org/wiki/Nim#Example_implementation

        We play normal (i.e. not misere) game: the player taking the last object wins

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.

        Returns
        -------
        move : list
            move[0] is the heap to take from (starts at 1)
            move[1] is the number of obj to take from heap #move[0]

        """
        if random.random() < self.epsilon:
            return self.randomMove(heaps)

        else:
            nim_sum = self.compute_nim_sum(heaps)
            if nim_sum == 0:
                # You will lose :(
                count_non_0 = sum(x > 0 for x in heaps)
                if count_non_0 == 0:
                    # Game is already finished, return a dumb move
                    move = [-1, -1]
                else:
                    # Take any possible move
                    move = [heaps.index(max(heaps)) + 1, 1]
                return move

            # Calc which move to make
            for index, heap in enumerate(heaps):
                target_size = heap ^ nim_sum
                if target_size < heap:
                    amount_to_remove = heap - target_size
                    move = [index + 1, amount_to_remove]
                    return move

                
class QL_Player(OptimalPlayer):
    def __init__(self, epsilon, player):
        super(QL_Player, self).__init__(epsilon = epsilon, player = player)
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
        self.qvals = Qvals
    def copy(self):
        new_player = QL_Player(self.epsilon, self.player)
        new_player.qvals = self.qvals.copy()
        return new_player
        
    def QL_Move(self, heaps):
        
        # choose A from state s with eps-greedy policy
        current_config = str(heaps[0]) + str(heaps[1]) + str(heaps[2])
        
        if random.random() < self.epsilon:
            i, j, k = random.choice(list(self.qvals[current_config]))
        else:
            #print(self.qvals[current_config])
            max_val = max(self.qvals[current_config].values())
            max_keys = []
            for key, value in self.qvals[current_config].items():
                if value == max_val:
                    max_keys.append(key)
            #print(max_keys)
            i, j, k = random.choice(max_keys) # choose at random amongst the best options 
        
        action = [int(i), int(j), int(k)]
        #print('action: ', action)
        nb_stick, pile_to_take = max(np.array(heaps) - np.array(action)), np.argmax(np.array(heaps) - np.array(action))
        move = [pile_to_take + 1, nb_stick]
        #print('move: ', move)
        #print(action)
        return move, action
    
    def update_qval(self, ql_action, other_move, heaps_before, heaps_after, env, alpha, gamma):
        """
        inputs:
                - ql_action: the last action by ql player
                - other_move: the move played just after 'ql_action'
                - heaps_before: state of the game before 'ql_action'
                - heaps_after: current game (after 'other_move')
        """
        #print(heaps_after)
        current_config = str(heaps_after[0]) + str(heaps_after[1]) + str(heaps_after[2])
        reward = env.reward(self.player)
        #print('player: ', self.player, 'current config: ', current_config, 'previous config: ', heaps_before, 'action: ', ql_action, 'other move: ', other_move, 'reward: ', reward)
        #reward = env.reward(env.current_player)
        if self.qvals[current_config]: # the dictionnary is not empty (ie we can take an action)
            max_val = max(self.qvals[current_config].values())
            max_keys = []
            for key, value in self.qvals[current_config].items():
                if value == max_val:
                    max_keys.append(key)
            Q_s_new_a = self.qvals[current_config][random.choice(max_keys)] # in the algo: max Q(S', a) over all a (on peut lui trouver un meilleur nom)
        else:
            Q_s_new_a = 0
        
        previous_config = str(heaps_before[0]) + str(heaps_before[1]) + str(heaps_before[2])
        
        # update Q(s, a)
        i, j, k = ql_action
        #print(self.qvals[previous_config], ql_action)
        #print(heaps_before, ql_action, other_move)
        #print('updating config ', previous_config, ' wtih action ', ql_action, ' for player ', self.player)
        self.qvals[previous_config][str(i) + str(j) + str(k)] = (1 - alpha) * self.qvals[previous_config][str(i) + str(j) + str(k)] + alpha * reward + gamma * alpha * Q_s_new_a
        #print(self.qvals['210'])
        #print(self.qvals[previous_config])
        
       
    def act(self, heaps, **kwargs):
        return self.QL_Move(heaps)
            
        