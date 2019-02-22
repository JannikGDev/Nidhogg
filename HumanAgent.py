import pygame
from Constants import *


class HumanAgent:

    def __init__(self):
        self.currentDir = NORTH

    def step(self, board):

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.currentDir = WEST
                if event.key == pygame.K_RIGHT:
                    self.currentDir = EAST
                if event.key == pygame.K_UP:
                    self.currentDir = NORTH
                if event.key == pygame.K_DOWN:
                    self.currentDir = SOUTH

        return self.currentDir
