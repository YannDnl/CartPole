# Model parameters

DT = 0.03  # time step
R = 0.5  # length of the pole (m)
G = 9.81  # acceleration due to gravity (m/s^2)
m = 10  # mass of the cart (kg)
M = 0.1  # mass of the pole (kg)
DURATION = 600  # duration of the simulation (ticks)

# PID control parameters

PROPORTIONAL_CORRECTION = 200
INTEGRAL_CORRECTION = 20
DERIVATIVE_CORRECTION = 80

### RL parameters

ACTION_LIST = [-1, 0, 1]
DQN_ACTION_SCALING = 200

# Learning parameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64

# Model hyperparameters
STATE_SIZE = 4  # Number of state variables
ACTION_SIZE = len(ACTION_LIST)  # Number of actions (left or right)
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
EPISODES = 1000

# Animation parameters

SCREEN_COLOR = (0, 0, 0)  # black
CART_COLOR = (150, 0, 255)  # blue
POLE_COLOR = (150, 150, 255)  # black
BUBBLE_COLOR = (50, 50, 50)  # grey
RAIL_COLOR = (255, 255, 255)  # white
SCREEN_WIDTH = 800  # width of the screen (pixels)
SCREEN_HEIGHT = 400  # height of the screen (pixels)
CART_WIDTH = 100  # width of the cart (pixels)
CART_HEIGHT = 50  # height of the cart (pixels)
POLE_LENGTH = R * 300 # length of the pole (pixels) (scaled for visualization)
FPS = 30  # frames per second
X_SCALE = 200  # scale factor for x position