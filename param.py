DT = 0.03  # time step
R = 0.5  # length of the pole (m)
G = 9.81  # acceleration due to gravity (m/s^2)
m = 10  # mass of the cart (kg)
M = 0.1  # mass of the pole (kg)

X_SCALE = 10000  # scale factor for x position

PROPORTIONAL_CORRECTION = 200
INTEGRAL_CORRECTION = 10
DERIVATIVE_CORRECTION = 30

SCREEN_WIDTH = 800  # width of the screen (pixels)
SCREEN_HEIGHT = 400  # height of the screen (pixels)
CART_WIDTH = 100  # width of the cart (pixels)
CART_HEIGHT = 50  # height of the cart (pixels)
POLE_LENGTH = R * 300 # length of the pole (pixels) (scaled for visualization)
FPS = 30  # frames per second
DURATION = 600  # duration of the simulation (frames)