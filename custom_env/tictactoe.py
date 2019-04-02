import numpy as np
import random
import copy

class TicTacToeEnv():
    """ Custom tictactoe environment based on openai gym format

    Attributes
    ----------
        symbols - possible symbols on board
        action_space (int)- number of possible actions
        observation_space (int) - number of possible observations
        state (numpy) - array representing the board (9 positions)
        on_move (int) - value represinting turn ('X' or 'O')

    Note
    ----
        AI always moves with 'X' 

    """


    def __init__(self):
        self.symbols = ['O', ' ', 'X'];
        self.action_space = 9
        self.observation_space = 9

    def reset(self):
        """ Reset the state to an empty board

        Returns
        -------
            state (numpy) - array representing current board
        """
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.on_move = 1
        return self.state

    def step(self, action):
        """ Change the state based on an action

        Parameters
        ----------
            action (discrete) - number 1-9 that represents board location 

        Returns
        -------
            state (numpy) - new state given action
            reward (int) - reward for action
            done (bool) - flag for game over
            {} (dict) - any potential extra information for user
        """

        done = False
        reward = 0
        
        current_val = self.state[action] # get current value at the location

        if (current_val != 0):  # position is not empty
            done = True
            reward = -10 
        else:
            self.state[action] = self.on_move # update board
            self.on_move = -self.on_move # switch to 'O'

            # check game statis after most recent 'X' move
            if self.isWin(self.state, -self.on_move): # if 'X' (AI) wins
                reward = 100
                done = True
            elif self.complete() == True: # tie
                reward = 10
                done = True
            else:
                randomMove = self.getMove() # either random or optimal move for 'O'
                self.state[randomMove] = self.on_move # update board with 'O'
                self.on_move = -self.on_move # change back to 'X'

                 # check game statis after most recent 'O' move
                if self.isWin(self.state, -self.on_move): # if 'O' wins (AI loses)
                    reward = -1
                    done = True
                elif self.complete(): # tie
                    reward = 10
                    done = True

        return self.state, reward, done, {}

    def render(self):
        """ Create a visual status of the board """
        print("on move: " , self.symbols[self.on_move+1]) # -1 is 'O', 1 is 'X'
        for i in range (9):
            print (self.symbols[self.state[i]+1], end=" ");
            if ((i % 3) == 2):
                print();

    def seed(self, seed):
        """ Seed the environment """
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed)
        self.seed = random.seed(seed)

    def isWin(self, board, marker):
        """ Check for a winning situation

        Parameters
        ----------
            board (numpy) - state of the board
            marker (char) - which turn for which we check for a win

        Return
        ------
            bool - true if winning scenario
        """
        for i in range(3):
            # check horizontals and verticals
            if ((board[i * 3] == marker and board[i * 3 + 1] == marker and board[i * 3 + 2] == marker) or (board[i + 0] == marker and board[i + 3] == marker and board[i + 6] == marker)):
                return True
        # check diagonals
        if((board[0] == marker and board[4] == marker and board[8] == marker) or (board[2] == marker and board[4] == marker and board[6] == marker)):
            return True
        return False

    def complete(self):
        """ Check for a tie """
        for i in range(9):
            if (self.state[i] == 0):
                return False
        return True

    def getMove(self):
        """ Return a move - either optimal or random

        Returns
        -------
            move (int) - integer position to place a marker on
        """

        # check if bot can win in the next move
        # for i in range(0,9):
        #     board_copy = copy.deepcopy(self.state)
        #     if board_copy[i] == 0:
        #         board_copy[i] = self.on_move
        #         if self.isWin(board_copy, self.on_move):
        #             return i

        # Block opponent
        # for i in range(0,len(self.state)):
        #     board_copy = copy.deepcopy(self.state)
        #     if board_copy[i] == 0:
        #         board_copy[i] = -self.on_move
        #         if self.isWin(board_copy, -self.on_move):
        #             return i

        # check for space in the corners, and take it
        # corners = [0,2,6,8]
        # move = self.chooseMove(corners)
        # if move != None:
        #     return move

        # If the middle is free, take it
        # if self.state[4] == 0:
        #     return 4

        # else, take one free space on the sides
        allMoves = [0,1,2,3,4,5,6,7,8]
        move = self.chooseMove(allMoves)
        if move != None:
            return move

    def chooseMove(self, moveList):
        """ Select a valid move from possible list 

        Parameters
        ----------
            moveList (list<int>) - list of possible board positions

        Returns
        -------
            random integer from possible moves, or None if no moves possible
        """

        poss_moves = []
        for val in moveList:
            if self.state[val] == 0:
                poss_moves.append(val)
        if len(poss_moves) != 0:
            return random.choice(poss_moves)
        else:
            return None

    def close(self):
        return

                
if __name__ == '__main__':
    env = TicTacToeEnv()
    env.reset()
    env.render()
    print(env.step(4))
    env.render()
    print(env.step(6))
    env.render()
