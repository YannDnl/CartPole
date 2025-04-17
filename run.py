import pygame
import numpy as np
import sys

from param import SCREEN_HEIGHT, SCREEN_WIDTH, CART_HEIGHT, CART_WIDTH, POLE_LENGTH, FPS, DURATION, X_SCALE
from environment import initialState, nextState
from agent import getAction

pygame.init()

screen_width = SCREEN_WIDTH
screen_height = SCREEN_HEIGHT
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Computer plays Cart-Pole")

white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)

cart_width = CART_WIDTH
cart_height = CART_HEIGHT
pole_length = POLE_LENGTH

def draw_cart_pole(x, theta):
    screen.fill(white)

    # Draw the cart
    cart_x = screen_width // 2 + int(x * X_SCALE)  # Scale the position
    pygame.draw.rect(screen, blue, (cart_x - cart_width // 2, 2 * screen_height//3 - cart_height, cart_width, cart_height))

    # Draw the pole
    pole_end_x = cart_x + pole_length * np.cos(theta)
    pole_end_y = 2 * screen_height//3 - cart_height // 2 - pole_length * np.sin(theta)
    pygame.draw.line(screen, black, (cart_x, 2 * screen_height//3 - cart_height // 2), (pole_end_x, pole_end_y), 5)


state, reward = initialState()
memory = (0, 0)  # old_reward, reward_integral

clock = pygame.time.Clock()
frame = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if frame < DURATION:
        action, memory = getAction(state, memory)
        state, reward = nextState(state, action)

        draw_cart_pole(state[0], state[2])
        frame += 1
    else:
        running = False

    pygame.display.flip()
    clock.tick(FPS)  # Limit the frame rate to 30 FPS

pygame.quit()
sys.exit()