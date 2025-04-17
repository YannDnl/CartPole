import pygame
import numpy as np
import sys

from param import POLE_COLOR, CART_COLOR, SCREEN_COLOR, BUBBLE_COLOR, RAIL_COLOR, SCREEN_HEIGHT, SCREEN_WIDTH, CART_HEIGHT, CART_WIDTH, POLE_LENGTH, FPS, DURATION, X_SCALE
from environment import initialState, nextState
from agent import getAction

pygame.init()

screen_width = SCREEN_WIDTH
screen_height = SCREEN_HEIGHT
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Computer plays Cart-Pole")

cart_width = CART_WIDTH
cart_height = CART_HEIGHT
pole_length = POLE_LENGTH

def draw_cart_pole(x, theta):
    screen.fill(SCREEN_COLOR)

    cart_x = screen_width // 2 + int(x * X_SCALE)  # Scale the position

    pole_start_x = cart_x
    pole_start_y = 2 * screen_height//3 - cart_height // 2
    pole_end_x = pole_length * np.cos(theta)
    pole_end_y = - pole_length * np.sin(theta)

    # Icon in the corner when the cart is out of the screen
    divison_factor = 3
    icon_x = pole_length // divison_factor + 25
    icon_y = icon_x
    icon_cart_w = cart_width // divison_factor
    icon_cart_h = cart_height // divison_factor
    icon_pole_start_x = icon_x + icon_cart_w // 2
    icon_pole_start_y = icon_y + icon_cart_h // 2
    pygame.draw.circle(screen, BUBBLE_COLOR, (icon_pole_start_x, icon_pole_start_y), icon_x - 15)
    pygame.draw.line(screen, RAIL_COLOR, (icon_pole_start_x - (icon_x - 15), icon_pole_start_y), (icon_pole_start_x + (icon_x - 15), icon_pole_start_y), 1)
    pygame.draw.rect(screen, CART_COLOR, (icon_x, icon_y, icon_cart_w, icon_cart_h))
    pygame.draw.line(screen, POLE_COLOR, (icon_pole_start_x, icon_pole_start_y), (icon_pole_start_x + pole_end_x//divison_factor, icon_pole_start_y + pole_end_y//divison_factor), 5)

    # Draw the cart's rail

    pygame.draw.line(screen, RAIL_COLOR, (0, 2 * screen_height//3 - cart_height // 2), (screen_width, 2 * screen_height//3 - cart_height // 2), 2)

    # Draw the cart
    pygame.draw.rect(screen, CART_COLOR, (cart_x - cart_width // 2, 2 * screen_height//3 - cart_height, cart_width, cart_height))

    # Draw the pole
    pygame.draw.line(screen, POLE_COLOR, (pole_start_x, pole_start_y), (pole_start_x + pole_end_x, pole_start_y + pole_end_y), 5)

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