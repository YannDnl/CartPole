# Cart pole problem

I have recreated the cart pole problem on a custom environment. I have implemented two methods to solve it. Control engineering through a PID corrector and a reinforcement learning agent (in progress).

## Environment
### Equations of motion

$$M\left(r\frac{d^2\theta}{dt^2} - \frac{d^2x}{dt^2}\sin\theta\right) = -Mg\cos\theta$$
$$M\left(\frac{d^2x}{dt^2}\cos\theta - r\left(\frac{d\theta}{dt}\right)^2\right) = -Mg\sin\theta - F$$
$$m\frac{d^2x}{dt^2} = f + F\cos\theta$$

Where $f$ is the control term. The state of the system being described by $s(t) = (\theta(t), \frac{d\theta}{dt}(t), x(t), \frac{dx}{dt}(t))$, we want to obtain $x$ and $\theta$ in $t + dt$.

Thus we can rewrite the equation in the following way:

$$r\left(\frac{d^2\theta}{dt^2} - \frac{M}{m}\sin\theta\cos\theta\left(\frac{d\theta}{dt}\right)^2\right) + g\cos\theta\left(1+\frac{M}{m}\sin^2\theta\right) - f\frac{\sin\theta}{m}\left(1 - \frac{M}{m}\cos^2\theta\right) + F\frac{M}{m^2}\cos^3\theta\sin\theta = 0$$

$$m\frac{d^2x}{dt^2} = f\left(1 - \frac{M}{m}\cos^2\theta\right) + M\cos\theta\left(r\left(\frac{d\theta}{dt}\right)^2 - g\sin\theta\right)$$

Now assuming $m \gg M$,

$$r\frac{d^2\theta}{dt^2} + g\cos\theta - f\frac{\sin\theta}{m} = 0$$

$$m\frac{d^2x}{dt^2} = f + M\cos\theta\left(r\left(\frac{d\theta}{dt}\right)^2 - g\sin\theta\right)$$

That we solve for $s(t+dt)$ using Euler's method.

### Animations

The Environment is animated using Pygame, the steps are computed dyanmically.

## Agent

### Control Engineering

We use a PID corrector to set the control $f$, ie:

$$f = k_p \epsilon + k_i \int_0^t \epsilon dt + k_d \frac{d\epsilon}{dt}$$

Where $\epsilon \in [-\pi, \pi[$ and $\epsilon \equiv \frac{\pi}{2} - \theta [2\pi]$.

Works well.

### Reinforcement learning

I am implementing a DQN algorithm to control the system.