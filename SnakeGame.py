import numpy as np
from SnakePlayer import SnakePlayer
import math
import random
import pygame
from pygame import Rect
import time
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)


FOOD = 1

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

COLORS = [
    (100, 100, 100),
    (255, 0, 0),
    (0, 255, 0),
    (0, 120, 0),
    (0, 0, 255),
    (0, 0, 120),
    (0, 255, 255),
    (0, 120, 120),
    (255, 255, 0),
    (120, 120, 0)
]


class SnakeGame:

    def __init__(self, player_count, width=15, height=15, render=False):
        self.player_count = player_count
        self.players = []
        self.board = np.zeros((width, height)).astype(np.int)
        self.ghost_board = np.zeros((width, height)).astype(np.int)
        self.render = render
        self.setup()
        self.running = True
        self.food = (0, 0)

        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    def step(self):

        for i in range(self.player_count):

            self.players[i].step(self.board)

        for i in range(self.player_count):
            self.check_alive(self.players[i])

        self.board = self.board * 0
        self.ghost_board = self.ghost_board * 0
        self.board[self.food[0], self.food[1]] = FOOD

        for i in range(self.player_count):
            self.board, self.ghost_board = self.players[i].draw_snake(self.board, self.ghost_board)

        if self.render:
            self.render_game()

        return self.running

    def setup(self):

        self.board = self.board*0
        self.players.clear()

        radius = math.floor(self.board.shape[0]*0.3)
        center = (self.board.shape[0]//2, self.board.shape[1]//2)

        color = 2
        for i in range(self.player_count):

            dx = math.cos(i*2*(math.pi/self.player_count))*radius
            dy = math.sin(i * 2 * (math.pi / self.player_count)) * radius

            self.players.append(SnakePlayer(color, (math.floor(center[0] + dx), math.floor(center[1] + dy))))

            self.board, self.ghost_board = self.players[i].draw_snake(self.board, self.ghost_board)

            color += 2

        empty_space = np.where(self.board == 0)
        food_spawn = random.randrange(0, len(empty_space[0]))

        self.food = (empty_space[0][food_spawn], empty_space[1][food_spawn])

        self.board[self.food[0], self.food[1]] = FOOD

    def check_alive(self, snake):

        if snake.alive is False:
            return

        if snake.head_pos[0] < 0 or snake.head_pos[0] >= 15 or snake.head_pos[1] < 0 or snake.head_pos[1] >= 15:
            snake.alive = False
            return

        for other in self.players:

            if other.alive and other.color is not snake.color:
                for part in other.parts:

                    if snake.head_pos[0] == part[0] and snake.head_pos[1] == part[1]:
                        snake.alive = False
                        return

    def render_game(self):

        self.screen.fill((0, 0, 0, 0))
        stepx = 40
        stepy = 40

        for x in range(0,15):
            for y in range(0, 15):
                self.screen.fill(COLORS[self.board[x][y]], rect=Rect(x*stepx, y*stepy, stepx, stepy))

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                self.running = False

        time.sleep(0.1)

        pygame.display.flip()


if __name__ == '__main__':

    game = SnakeGame(player_count=4, render=True)

    running = True

    while running:

        running = game.step()
