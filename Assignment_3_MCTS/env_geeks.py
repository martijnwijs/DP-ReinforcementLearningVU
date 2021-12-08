
# Tic-Tac-Toe Program using
# random number in Python

# part of the code for this environment is copied from 
# https://www.geeksforgeeks.org/python-implementation-automatic-tic-tac-toe-game-using-random-number/

# importing all necessary libraries
import numpy as np
import random
from time import sleep
import copy

class environment:
    def __init__(self):
        return

    def init_board(self, start_pos = False):
        if start_pos == False:
            return (np.array([[0, 0, 0],  # 8initializes an empty board
                              [0, 0, 0],
                              [0, 0, 0]]))
        else:
            return (np.array([[0, 2, 0],  # 8initializes an given board
                              [1, 1, 2],
                              [0, 0, 0]]))
    # Check for empty places on board 
    def possibilities(self,board): ##################### 
        l = []
        for i in range(len(board)):
            for j in range(len(board)):
                
                if board[i][j] == 0:
                    l.append((i, j))
        return(l)
    
    # Select a random place for the player
    def random_place(self, board, player):
        selection = self.possibilities(board)
        current_loc = random.choice(selection)
        board[current_loc] = player
        return(board)

    def move(self, position, player, boardstate):
        #print("position:", position)
        board = copy.deepcopy(boardstate)
        #print(board[(1,1)])
        board[position] = player
        #print("move from env: ", board)
        return board

    # Checks whether the player has three 
    # of their marks in a horizontal row
    def row_win(self, board, player):
        for x in range(len(board)):
            win = True
            
            for y in range(len(board)):
                if board[x, y] != player:
                    win = False
                    continue
                    
            if win == True:
                return(win)
        return(win)
    
    # Checks whether the player has three
    # of their marks in a vertical row
    def col_win(self, board, player):
        for x in range(len(board)):
            win = True
            
            for y in range(len(board)):
                if board[y][x] != player:
                    win = False
                    continue
                    
            if win == True:
                return(win)
        return(win)
    
    # Checks whether the player has three
    # of their marks in a diagonal row
    def diag_win(self, board, player):
        win = True
        y = 0
        for x in range(len(board)):
            if board[x, x] != player:
                win = False
        if win:
            return win
        win = True
        if win:
            for x in range(len(board)):
                y = len(board) - 1 - x
                if board[x, y] != player:
                    win = False
        return win
    
    # Evaluates whether there is
    # a winner or a tie 
    def evaluate(self, board):
        winner = 0
        
        for player in [1, 2]:
            if (self.row_win(board, player) or
                self.col_win(board,player) or 
                self.diag_win(board,player)):
                    
                winner = player
                
        if np.all(board != 0) and winner == 0:
            winner = -1
        return winner


    def play_game(self, boardstate):
        '''random gameplay'''
        board = copy.deepcopy(boardstate)
        winner, counter = 0, 1
        #print(board)
        winner = self.evaluate(board)
        
        while winner == 0:
            for player in [1, 2]: # player one is MCTS agent, player 2 is random agent
                board = self.random_place(board, player)     
                counter += 1
                winner = self.evaluate(board)
                if winner != 0:
                    break
        #print("boardstate final move of rollout: ")
        #print(board)
        return(winner)
    
    def check_winner(self, board):
        winner = self.evaluate(board)
        return winner

# Driver Code
#print("Winner is: " + str(play_game()))