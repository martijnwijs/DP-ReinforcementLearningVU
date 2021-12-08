from env_geeks import environment
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class TreeNode():
    '''node that represents a board state'''
    def __init__(self, state, parent, player, visits=0.):
        
        self.parent = parent
        self.visits = visits
        self.ucb = 10.**9 
        self.children = [] 
        self.score = 0.
        self.wins = 0.
        self.state = state
        self.id = random.random()
        self.player = player
        self.avg_score = 0.
        self.ch_winning = 0.
        
    def calc_ucb(self): # function that returns the ucb value of a node
        self.ucb = 0.00001*self.score/(self.visits+5) + math.sqrt(2)*(math.sqrt(math.log(self.parent.visits+1)/self.visits)) # +1 makes sure the exploration part doesnt become 0 at the first expansion, otherwise it will never be explored anymore
        self.avg_score = self.score / self.visits
        self.ch_winning = self.wins / self.visits
        return 

class MCTS():
    def __init__(self, n_runs, environment, c=2, strategy="maximin", start_pos=False): # c is tradeoff between exploration and exploitation
        self.n_runs = n_runs
        self.score_sim = 0 # score used at each simulation
        self.env = environment
        self.current_node = TreeNode(state = self.env.init_board(start_pos),parent=None, player = 1) # 1 player 1, -1 player 2
        self.winner = 0 # 0 is draw, 1 is player 1, 2 is player 2
        self.win = 0 # variable that checks if player 1 has won
        self.break_simulation = False # variable used when the whole tree is searched
        self.strategy = strategy
        self.q_values = {} 

    def select(self): # go to a leafnode by ucb selection
        while self.current_node.visits > 0 and self.break_simulation == False:
            if self.env.evaluate(self.current_node.state) != 0: # check if game is already finished
                break

            if len(self.current_node.children) == 0: #  no children
                self.expansion() # expand branch
            else:
                self.current_node = max(self.current_node.children, key = lambda child: child.ucb) # chose child with highest ucb
        self.current_node.visits += 1
        return

    def expansion(self):
        if len(self.env.possibilities(self.current_node.state)) != 0: # check if game is not already finished
            for position in self.env.possibilities(self.current_node.state):
                nextstate = self.env.move(position, self.current_node.player, self.current_node.state) 
                self.current_node.children.append(TreeNode(nextstate, parent=self.current_node, player = self.next_player()))
        else:
            self.break_simulation = True
        return

    def simulation(self):
        self.winner = self.env.play_game(self.current_node.state)
        self.evaluate_score() 
        return

    def leafnode_evaluation(self, nodestate):
        winner = self.env.evaluate(nodestate)
        if winner ==1: 
            score = 1
        elif winner == 2:
            score = -1
        elif winner == -1: 
            score = 0
        return score

    def evaluate_score(self):
        if self.winner == 1:
            self.score_sim = 1
            self.win = 1
        elif self.winner == 2:
            self.score_sim = -1
            self.win = 0
        elif self.winner == -1:
            self.score_sim = 0
            self.win = 0
        return

    def next_player(self):
        '''function to switch players per node'''
        if self.current_node.player == 1:
            return 2
        else:
            return 1

    def backpropagation(self):
        i = 0
        backprop = True
        while backprop == True: # update the values
            if self.current_node.parent == None:
                backprop = False
            if self.current_node.player == 1:  
                self.current_node.score += self.score_sim   
                self.current_node.wins += self.win
            else:
                self.current_node.score += self.score_sim
                self.current_node.wins += self.win
            if i>0:
                self.current_node.visits += 1
            if backprop == True:
                self.current_node.calc_ucb()
                self.current_node = self.current_node.parent      
            i += 1
         
        return 

    def move(self):
        if self.strategy == "maximin":
            # find min node of children
            maximin_val = []
            for child in self.current_node.children:
                try: 
                    maximin_val.append(min(child.children, key = lambda child: child.avg_score).avg_score)
                except: # two options possible: 1) this branch is not explored , because it is not a preferable branch . therefore give a negative reward of -1
                    # 2) this branch is a leafnode, therefore first check for this
                    if self.env.evaluate(child.state) != 0: # game finished
                        maximin_val.append(self.leafnode_evaluation(child.state))
                    else: # not a good branch 
                        maximin_val.append(-1)
            ind = np.argmax(np.array(maximin_val))
            self.current_node = self.current_node.children[ind] # move to child with maximin score
            self.current_node.parent = None # remove parent node   
            return
        
        elif self.strategy == "random": # move random
            self.current_node = max(self.current_node.children, key = lambda child: child.score) # move to child with highest score
            self.current_node.parent = None # remove parent node 
            return
    
    def store_q_values(self, iteration):
        '''stores the q values for plotting convergence''' 
        if self.strategy == "random":
            for child in self.current_node.children:
                winning_chance = child.ch_winning
                if child in self.q_values and child.visits > 0:
                    self.q_values[child].append(winning_chance)
                        
                elif child.visits > 0:
                    self.q_values[child] = [(winning_chance)]
   
        if self.strategy == "maximin" and iteration > 500: # bigger then 500 because otherwise there are unexplored nodes and the v values cannot be stored
            for child in self.current_node.children:
                try: 
                    winning_chance = min(child.children, key = lambda child: child.avg_score).ch_winning
                    
                except: # two options possible: 1) this branch is not explored , because its parent is not a preferable branch . therefore give a negative reward of -1
                    # 2) this branch is a leafnode, therefore first check for this
                    if self.env.evaluate(child.state) != 0: # game finished
                        maximin_v = self.leafnode_evaluation(child.state)
                
                        if maximin_v == 1:
                            winning_chance = maximin_v
                        else: winning_chance = 0.
                if child in self.q_values and child.visits > 0:
                    self.q_values[child].append(winning_chance)
                        
                elif child.visits > 0:
                    self.q_values[child] = [winning_chance]
        return

    def move_other_player(self):
        if self.strategy == "maximin":
            maximin_val = []
            for child in self.current_node.children:
                try: 
                    maximin_val.append(max(child.children, key = lambda child: child.avg_score).avg_score)
                except: # two options possible: 1) this branch is not explored , because its parent is not a preferable branch . therefore give a negative reward of -1
                    # 2) this branch is a leafnode, therefore first check for this
                    if self.env.evaluate(child.state) != 0: # game finished
                        maximin_val.append(self.leafnode_evaluation(child.state))
                    else: # not a good branch 
                        maximin_val.append(-1)
            ind = np.argmin(np.array(maximin_val))
            self.current_node = self.current_node.children[ind] # move to child with maximin score
            self.current_node.parent = None # remove parent node   
            
        elif self.strategy == "random":
            try:
                self.current_node = random.choice(self.current_node.children)
            except:
                for position in self.env.possibilities(self.current_node.state):
                    nextstate = self.env.move(position, self.current_node.player, self.current_node.state) 
                    self.current_node.children.append(TreeNode(nextstate, parent=self.current_node, player = self.next_player()))
                self.current_node = random.choice(self.current_node.children)
            self.current_node.parent = None
            self.current_node.children = []           
        
        return 

    def debug(self):
        for move in self.current_node.children:
            print(move.state)
            print("score", move.score)
            print("n visits", move.visits)
            print("visits parent", move.parent.visits)
            print("ucb", move.ucb)
            pass
    
    def plot_q_values(self):
        for action in self.q_values:
            plt.plot(np.linspace(0, len(self.q_values[action]), len(self.q_values[action])), self.q_values[action], label = str(action.state))
        plt.legend()
        plt.xlabel("MC simulation")
        plt.ylabel("chance of winning")
        plt.show()
        return

    def run(self, plot_q=True):
        self.q_values = {}
        for run in range(self.n_runs):
            self.select()
            self.simulation()
            self.backpropagation()
            if plot_q == True:
                self.store_q_values(run)
                
            self.break_simulation = False
        if plot_q == True:
            self.plot_q_values()
        return 

def game(n_runs, strategy, start_pos, plot_q=True):
    env = environment()
    #print(env)
    mcts = MCTS(c=2, n_runs=n_runs, environment = env, strategy = strategy,  start_pos = start_pos) # change starting position and strategy here
    while True:
        mcts.run(plot_q)
        mcts.move()
        if mcts.env.check_winner(mcts.current_node.state) != 0: # check for win:
            print("winner is: ", mcts.env.check_winner(mcts.current_node.state))
            break

        mcts.move_other_player()
        if mcts.env.check_winner(mcts.current_node.state) != 0: # check for win:
            print("winner is: ", mcts.env.check_winner(mcts.current_node.state))
            break
    return mcts.env.check_winner(mcts.current_node.state)



# different simulations

# play game from given state by assignment using random strategy, report the chance of winning and convergence
winner = game(n_runs = 5000, strategy = "random", start_pos = True, plot_q=True)
print("winner random strategy from assignment position: ", winner)

# play game from given state by assignment using maximin strategy, report the chance of winning and convergence
winner = game(n_runs = 5000, strategy = "maximin", start_pos = True, plot_q=True)
print("winner maximin strategy from assignment position: ", winner)

# play game from initial state by random strategy, report convergence
winner = game(n_runs = 10000, strategy = "random", start_pos = False, plot_q=True)
print("winner random strategy from empty position: ", winner)

# play game from initial state by maximin strategy, report convergence
winner = game(n_runs = 40000, strategy = "maximin", start_pos = False, plot_q = True)
print("winner maximin strategy from empty position: ", winner)

print("play random strategy 10 times and maximin 10 times to see if perfect gameplay")
# play game 10 times using random strategy, report wins
winners = []
for i in range(10): # play 10 games
    winner = game(n_runs = 10000, strategy = "random", start_pos = False, plot_q = False)
    winners.append(winner)
    print("winner of this game is: ", winner)
print("the winner of 10 random opponent games (1 is player 1, 2 player 2, -1 draw): ", winners)

# play game from initial state by maximin strategy, report convergence 
# play game 10 times using maximin strategy, report wins
winners = []
for i in range(10): # play 10 games
    winner = game(n_runs = 40000, strategy = "maximin", start_pos = False, plot_q = False)
    winners.append(winner)
    print("winner of this game is: ", winner)
print("the winner of 10 maximin opponent games (1 is player 1, 2 player 2, -1 draw): ", winners)
