import numpy as np


MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_STRAIGHT = 2

NORTH = 0
WEST = 1
SOUTH = 2
EAST = 3


class SnakePlayer:

    def __init__(self, color, start_pos, start_dir=SOUTH, start_length=3):

        self.life = 100
        self.color = color
        self.direction = start_dir
        self.alive = True

        self.head_pos = start_pos
        self.parts = []

        for i in range(start_length):

            if self.direction == SOUTH:
                self.parts.append((start_pos[0], start_pos[1] + (i+1)))
            elif self.direction == NORTH:
                self.parts.append((start_pos[0], start_pos[1] - (i+1)))
            elif self.direction == WEST:
                self.parts.append((start_pos[0] - (i+1), start_pos[1]))
            elif self.direction == EAST:
                self.parts.append((start_pos[0] + (i+1), start_pos[1]))
            else:
                ValueError()

    def step(self, board):

        if self.alive is False:
            return

        self.direction = SOUTH

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

    def draw_snake(self, board, ghost_board):

        if self.alive:
            board[self.head_pos[0], self.head_pos[1]] = self.color

            for part in self.parts:
                board[part[0], part[1]] = self.color + 1

        return board, ghost_board

