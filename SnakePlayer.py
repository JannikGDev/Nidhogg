import numpy as np
from SnakeAgent import SnakeAgent
from Constants import *
from HumanAgent import HumanAgent


class SnakePlayer:

    def __init__(self, agent, color, start_pos, start_dir=SOUTH, start_length=4):

        self.life = 100
        self.color = color
        self.direction = start_dir
        self.alive = True
        self.done = False
        self.reward = 0

        self.agent = SnakeAgent(agent, color)
        self.human = HumanAgent()
        self.head_pos = start_pos
        self.parts = []

        for i in range(start_length):

            if self.direction == SOUTH:
                self.parts.append((start_pos[0], start_pos[1]))
            elif self.direction == NORTH:
                self.parts.append((start_pos[0], start_pos[1]))
            elif self.direction == WEST:
                self.parts.append((start_pos[0], start_pos[1]))
            elif self.direction == EAST:
                self.parts.append((start_pos[0], start_pos[1]))
            else:
                ValueError()

    def step(self, board):

        if self.alive is False and self.done is True:
            return
        elif self.alive is False:
            self.done = True
            self.agent.step(board, -10, self.done)
            return

        self.life -= 1

        self.reward += -0.01

        self.direction = self.agent.step(board, self.reward, self.done)

        amount = len(self.parts)
        for n in range(amount):
            if n == amount - 1:
                self.parts[amount - n - 1] = self.head_pos
            else:
                self.parts[amount - n - 1] = self.parts[amount - n - 2]

        if self.direction == SOUTH:
            self.head_pos = (self.head_pos[0], self.head_pos[1]+1)
        elif self.direction == NORTH:
            self.head_pos = (self.head_pos[0], self.head_pos[1] - 1)
        elif self.direction == WEST:
            self.head_pos = (self.head_pos[0] - 1, self.head_pos[1])
        elif self.direction == EAST:
            self.head_pos = (self.head_pos[0] + 1, self.head_pos[1])
        else:
            ValueError()

    def add_part(self):

        last_part = self.parts[len(self.parts)-1]

        self.parts.append((last_part[0], last_part[1]))

    def draw_snake(self, board, ghost_board):

        if self.alive:
            for part in self.parts:
                board[part[0], part[1]] = self.color + 1

            board[self.head_pos[0], self.head_pos[1]] = self.color
        else:
            for part in self.parts:
                if part[0] >= 0 and part[0] < MAP_SIZE and part[1] >= 0 and part[1] < MAP_SIZE:
                    ghost_board[part[0], part[1]] = self.color + 1

            if self.head_pos[0] >= 0 and self.head_pos[0] < MAP_SIZE and self.head_pos[1] >= 0 and self.head_pos[1] < MAP_SIZE:
                ghost_board[self.head_pos[0], self.head_pos[1]] = self.color

        return board, ghost_board

