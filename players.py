import random
import time
import pygame
import math
from connect4 import connect4
from copy import deepcopy
import numpy as np

class connect4Player(object):
	def __init__(self, position, seed=0, CVDMode=False):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)
		if CVDMode:
			global P1COLOR
			global P2COLOR
			P1COLOR = (227, 60, 239)
			P2COLOR = (0, 255, 0)

	def play(self, env: connect4, move: list) -> None:
		move = [-1]

class human(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		move[:] = [int(input('Select next move: '))]
		while True:
			if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
				break
			move[:] = [int(input('Index invalid. Select next move: '))]

class human2(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move[:] = [col]
					done = True

class randomAI(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move[:] = [random.choice(indices)]

class stupidAI(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move[:] = [3]
		elif 2 in indices:
			move[:] = [2]
		elif 1 in indices:
			move[:] = [1]
		elif 5 in indices:
			move[:] = [5]
		elif 6 in indices:
			move[:] = [6]
		else:
			move[:] = [0]

class minimaxAI(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		copy_env = deepcopy(env)
		env.visualize = False

		best_score, best_move = self.minimax(copy_env, 2, True)
		if best_move is not None:
			move[:] = [best_move]  # Apply the best move
		else:
			print("move not found")

	def minimax(self, copy_env, depth, isMax, last_move = None, last_player = None):
		
		opponent_position = 1 if self.position == 2 else 2  # Determine opponent's position based on AI's position
		
		if last_move and last_player and copy_env.gameOver(last_move, last_player):
    		# Assuming gameOver checks if the last move won the game
				if last_player == self.position:  # Win for the current AI
					return math.inf, None
				else:  # Loss for the current AI
					return -math.inf, None
		# No logic for handling draws at the moment

		if depth == 0:
			return self.evaluate(copy_env), None  
		
		best_move = None

		possible = copy_env.topPosition >= 0
		
		if isMax:
			maxEval = -math.inf
			for i, p in enumerate(possible):
				if p:
					new_env = deepcopy(copy_env)
					self.simulateMove(new_env, i, self.position)
					eval, _ = self.minimax(new_env, depth - 1, False, i, self.position)
					if eval > maxEval:
						maxEval = eval
						best_move = i  # Update the best move
			return maxEval, best_move
		else:
			minEval = math.inf
			for i, p in enumerate(possible):
				if p:
					new_env = deepcopy(copy_env)
					self.simulateMove(new_env, i, opponent_position)
					eval, _ = self.minimax(new_env, depth - 1, True, i, opponent_position)
					if eval < minEval:
						minEval = eval
						best_move = i  # Update the best move
			return minEval, best_move


	def simulateMove(self, env: connect4, move: int, player: int):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[0].append(move)

	def evaluate(self, env):
		# Scores for sequences
		score_3 = 6  # For sequences of 3 with an open end
		score_2 = 5    # For sequences of 2 with an open end
		score_1 = 1    # For single pieces with an open end

		def count_sequences(board, player, length):
			score = 0

			# Check for horizontal sequences with open ends
			for row in range(env.shape[0]):
				for col in range(env.shape[1] - length + 1):
					if all(board[row][col + i] == player for i in range(length)):
						if (col - 1 >= 0 and board[row][col - 1] == 0) or (col + length < env.shape[1] and board[row][col + length] == 0):
							score += 1

			# Check for vertical sequences with open ends
			for row in range(env.shape[0] - length + 1):
				for col in range(env.shape[1]):
					if all(board[row + i][col] == player for i in range(length)):
						if (row - 1 >= 0 and board[row - 1][col] == 0) or (row + length < env.shape[0] and board[row + length][col] == 0):
							score += 1

			# Check for diagonal sequences with open ends (down-right and up-right)
			for row in range(env.shape[0] - length + 1):
				for col in range(env.shape[1] - length + 1):
					# Down-right diagonal
					if all(board[row + i][col + i] == player for i in range(length)):
						if ((row - 1 >= 0 and col - 1 >= 0 and board[row - 1][col - 1] == 0) or
							(row + length < env.shape[0] and col + length < env.shape[1] and board[row + length][col + length] == 0)):
							score += 1
					# Up-right diagonal
					if row >= length - 1 and all(board[row - i][col + i] == player for i in range(length)):
						if ((row + 1 < env.shape[0] and col - 1 >= 0 and board[row + 1][col - 1] == 0) or
							(row - length >= 0 and col + length < env.shape[1] and board[row - length][col + length] == 0)):
							score += 1

			return score

		# Only sequences with open ends are considered for scoring
		my_score = sum(count_sequences(env.board, self.position, l) * (score_3 if l == 3 else score_2 if l == 2 else score_1) for l in [1, 2, 3])
		opponent_score = sum(count_sequences(env.board, self.opponent.position, l) * (score_3 if l == 3 else score_2 if l == 2 else score_1) for l in [1, 2, 3])
		
		return my_score - opponent_score
				
class alphaBetaAI(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		copy_env = deepcopy(env)
		env.visualize = False
		depth = 2

		# depth is always 2 for player1 and 3 for player 2 until move 10
		if np.count_nonzero(env.board) < 10:
			depth = 3
		else:
			depth = 2
		
		# initial call to minimax
		best_score, best_move = self.minimax(copy_env, depth, -math.inf, math.inf, True)
		if best_move is not None:
			move[:] = [best_move]  # Apply the best move
	
	def get_sorted_moves(self, last_move):
		board_size = 7
		# If last_move is at the extreme left (0), return moves in ascending order
		if last_move == 0:
			return list(range(board_size))
		
		# If last_move is None, default to the middle of the board
		if last_move is None:
			last_move = board_size // 2

		# Generate all possible moves in order from 0 to board_size - 1
		all_moves = list(range(board_size))
		# Sort moves based on their distance from the last move
		sorted_moves = sorted(all_moves, key=lambda x: abs(x - last_move))
		return sorted_moves

	def minimax(self, copy_env, depth, alpha, beta, isMax, last_move = None, last_player = None):
		
		if last_move and last_player and copy_env.gameOver(last_move, last_player):
				# Win
				if last_player == self.position:  
					return math.inf, None
				# Loss
				else: 
					return -math.inf, None

		# player 1 eval function
		if depth == 0 and self.position == 1:
			return self.evaluate1(copy_env), None

		# player 2 eval function
		if depth == 0 and self.position == 2:
			return self.evaluate2(copy_env), None
		
		best_move = None

		# Create move order based on last move played
		move_order = self.get_sorted_moves(last_move)

		if isMax:
			maxEval = -math.inf
			for i in move_order:
				if copy_env.topPosition[i] >= 0:
					new_env = deepcopy(copy_env)
					self.simulateMove(new_env, i, self.position)
					eval, _ = self.minimax(new_env, depth - 1, alpha, beta, False, i, self.position)
					if eval > maxEval:
						maxEval = eval
						best_move = i  # Update the best move
					alpha = max(alpha, eval)
					if alpha >= beta:  # Prune
						break
			return maxEval, best_move
		else:
			minEval = math.inf
			for i in move_order:
				if copy_env.topPosition[i] >= 0:
					new_env = deepcopy(copy_env)
					self.simulateMove(new_env, i, self.opponent.position)
					eval, _ = self.minimax(new_env, depth - 1, alpha, beta, True, i, self.opponent.position)
					if eval < minEval:
						minEval = eval
						best_move = i # Update the best move
					beta = min(beta, eval)
					if alpha >= beta: # Prune
						break
			return minEval, best_move

	def simulateMove(self, env: connect4, move: int, player: int):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[0].append(move)

	# Player 1 eval
	def evaluate1(self, env):
		score_3 = 6  
		score_2 = 5    
		score_1 = 1
		dic_row = {0: 2, 1: 1.6, 2: 1.2, 3: 0.8, 4: 0.4, 5: 0.4}
		def count_sequences(board, player, length):
			score = 0

			# Check for horizontal sequences with open ends
			for row in range(env.shape[0]):
				for col in range(env.shape[1] - length + 1):
					if all(board[row][col + i] == player for i in range(length)):
						if (col - 1 >= 0 and board[row][col - 1] == 0) and (col + length < env.shape[1] and board[row][col + length] == 0):
							score += dic_row[row] * 2
						elif (col - 1 >= 0 and board[row][col - 1] == 0) or (col + length < env.shape[1] and board[row][col + length] == 0):
							score += dic_row[row]

			# Check for vertical sequences with open ends
			for row in range(env.shape[0] - length + 1):
				for col in range(env.shape[1]):
					if all(board[row + i][col] == player for i in range(length)):
						if (row - 1 >= 0 and board[row - 1][col] == 0) or (row + length < env.shape[0] and board[row + length][col] == 0):
							score += 1.05

			# Check for diagonal sequences with open ends (down-right and up-right)
			if np.count_nonzero(env.board) >= 15:
				for row in range(env.shape[0] - length + 1):
					for col in range(env.shape[1] - length + 1):
						# Down-right diagonal
						if all(board[row + i][col + i] == player for i in range(length)):
							if ((row - 1 >= 0 and col - 1 >= 0 and board[row - 1][col - 1] == 0) or
								(row + length < env.shape[0] and col + length < env.shape[1] and board[row + length][col + length] == 0)):
								score += 1.2
						# Up-right diagonal
						if row >= length - 1 and all(board[row - i][col + i] == player for i in range(length)):
							if ((row + 1 < env.shape[0] and col - 1 >= 0 and board[row + 1][col - 1] == 0) or
								(row - length >= 0 and col + length < env.shape[1] and board[row - length][col + length] == 0)):
								score += 1.2

			return score

		my_score = sum(count_sequences(env.board, self.position, l) * (score_3 if l == 3 else score_2 if l == 2 else score_1) for l in [1, 2, 3])
		opponent_score = sum(count_sequences(env.board, self.opponent.position, l) * (score_3 if l == 3 else score_2 if l == 2 else score_1) for l in [1, 2, 3])
		
		return my_score - opponent_score

	# Player 2 eval
	def evaluate2(self, env):
		score_3 = 6  
		score_2 = 5    
		score_1 = 1
		dic_row = {0: 2.2, 1: 1.8, 2: 1.4, 3: 0.6, 4: 0.2, 5: 0.2}
		def count_sequences(board, player, length):
			score = 0

			# Check for horizontal sequences with open ends
			for row in range(env.shape[0]):
				for col in range(env.shape[1] - length + 1):
					if all(board[row][col + i] == player for i in range(length)):
						if (col - 1 >= 0 and board[row][col - 1] == 0) and (col + length < env.shape[1] and board[row][col + length] == 0):
							score += dic_row[row] * 2
						elif (col - 1 >= 0 and board[row][col - 1] == 0) or (col + length < env.shape[1] and board[row][col + length] == 0):
							score += dic_row[row]

			# Check for vertical sequences with open ends
			if np.count_nonzero(env.board) >= 8:
				for row in range(env.shape[0] - length + 1):
					for col in range(env.shape[1]):
						if all(board[row + i][col] == player for i in range(length)):
							if (row - 1 >= 0 and board[row - 1][col] == 0) or (row + length < env.shape[0] and board[row + length][col] == 0):
								score += 0.9

			# Check for diagonal sequences with open ends (down-right and up-right)
			if np.count_nonzero(env.board) >= 15:
				for row in range(env.shape[0] - length + 1):
					for col in range(env.shape[1] - length + 1):
						# Down-right diagonal
						if all(board[row + i][col + i] == player for i in range(length)):
							if ((row - 1 >= 0 and col - 1 >= 0 and board[row - 1][col - 1] == 0) or
								(row + length < env.shape[0] and col + length < env.shape[1] and board[row + length][col + length] == 0)):
								score += 1
						# Up-right diagonal
						if row >= length - 1 and all(board[row - i][col + i] == player for i in range(length)):
							if ((row + 1 < env.shape[0] and col - 1 >= 0 and board[row + 1][col - 1] == 0) or
								(row - length >= 0 and col + length < env.shape[1] and board[row - length][col + length] == 0)):
								score += 1

			return score

		my_score = sum(count_sequences(env.board, self.position, l) * (score_3 if l == 3 else score_2 if l == 2 else score_1) for l in [1, 2, 3])
		opponent_score = sum(count_sequences(env.board, self.opponent.position, l) * (score_3 if l == 3 else score_2 if l == 2 else score_1) for l in [1, 2, 3])
		
		return my_score - opponent_score

SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)




