import numpy as np
from SnakePlayer import SnakePlayer
import math
import random
import pygame
from pygame import Rect
import time
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
from Constants import *
from DQNAgent import DQNAgent

class SnakeGame:

    def __init__(self, player_count, width=MAP_SIZE, height=MAP_SIZE, render=False):
        self.player_count = player_count
        self.players = []
        self.board = np.zeros((width, height)).astype(np.int)
        self.ghost_board = np.zeros((width, height)).astype(np.int)
        self.render = render

        self.running = True
        self.food = [(0, 0)]
        self.plays = 0

        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA, 32)

        self.agent = DQNAgent((MAP_SIZE, MAP_SIZE, 4), 3)

        self.setup()

    def step(self):

        all_died = True
        for player in self.players:
            if player.alive:
                all_died = False



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

        if all_died:
            self.agent.replay(100)
            self.setup()

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

            self.players.append(SnakePlayer(self.agent, color, (math.floor(center[0] + dx), math.floor(center[1] + dy))))

            self.board, self.ghost_board = self.players[i].draw_snake(self.board, self.ghost_board)

            color += 2

        self.food.clear()
        for f in range(FOOD_AMOUNT):
            empty_space = np.where(self.board == 0)
            food_spawn = random.randrange(0, len(empty_space[0]))

            self.food.append((empty_space[0][food_spawn], empty_space[1][food_spawn]))

            self.board[self.food[f][0], self.food[f][1]] = FOOD

        self.plays += 1
        if self.plays % 100 == 0:
            self.render = True
        else:
            self.render = False

    def check_alive(self, snake):

        if snake.alive is False:
            return

        if snake.head_pos[0] < 0 or snake.head_pos[0] >= MAP_SIZE or snake.head_pos[1] < 0 or snake.head_pos[1] >= MAP_SIZE:
            snake.alive = False
            return

        if snake.life <= 0:
            snake.alive = False
            return

        for f in self.food:
            if snake.head_pos[0] == f[0] and snake.head_pos[1] == f[1]:
                empty_space = np.where(self.board == 0)
                food_spawn = random.randrange(0, len(empty_space[0]))

                new_food = (empty_space[0][food_spawn], empty_space[1][food_spawn])

                self.board[new_food[0], new_food[1]] = FOOD

                self.food.remove(f)
                self.food.append(new_food)

                snake.life = 100

                snake.add_part()

        for other in self.players:

            if other.alive:

                if snake.head_pos[0] == other.head_pos[0] and snake.head_pos[1] == other.head_pos[1] \
                        and other.color is not snake.color:
                    snake.alive = False
                    return

                for part in other.parts:

                    if snake.head_pos[0] == part[0] and snake.head_pos[1] == part[1]:
                        snake.alive = False
                        return

    def render_game(self):

        self.screen.fill((0, 0, 0, 0))
        stepx = math.floor(SCREEN_WIDTH / MAP_SIZE)
        stepy = math.floor(SCREEN_HEIGHT / MAP_SIZE)

        for x in range(0, MAP_SIZE):
            for y in range(0, MAP_SIZE):

                self.screen.fill(COLORS[self.board[x][y]], rect=Rect(x*stepx, y*stepy, stepx, stepy))

                if self.board[x][y] == EMPTY and self.ghost_board[x][y] != EMPTY:
                    self.screen.fill(GHOST_COLORS[self.ghost_board[x][y]], rect=Rect(x * stepx, y * stepy, stepx, stepy))

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
