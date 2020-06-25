---
title: "Temperature Control of Water Heater"
date: 2020-07-02T00:00:00+02:00
categories:
  - blog
tags:
  - Process Simulation
  - Process Control
  - Household
published: false
---

Have you ever wondered, how the control of the water heater in your home works? In Germany, there are two types of water heaters: 1) flow heaters (<i>Durchlauferhitzer</i>), where water will be heated up as it flows out of the water tap, or 2) water boilers (<i>Warmwasserspeicher</i>), where warm water is stored in a tank continuously.

Out of curiousity, I want to present the functionality and the power consumption of such water heaters in the following simulation. As a model, I will take the water boiler and consider the following process. The tank of the water boiler has a volume of $$V = 10\,L$$. The heating process is electrical with the heat flow $$Q$$. We will assume that the water is used continuously, meaning the inflow and outflow of the water is constant with $$F = 0.15\,\frac{L}{s}$$. The temperature of the water outlet should be at $$42^\circ{}C$$. At the beginning, the tank is already filled with water with the same temperature as the inlet $$T_\alpha = 20^\circ{}C$$.

#### Nomeclature
- $$U$$ : Internal energy      
- $$H$$ : Enthalpy flow rate      
- $$F$$ : Water flow rate in $$m^3\,s^{-1}$$      
- $$V$$ : Tank volume in $$m^3$$      
- $$T$$ : Temperature in $$K$$      
- $$Q$$ : Heat flow rate in $$W$$      
- $$\rho$$ : Water density in $$kg\,m^{-3}$$      
- $$c_p$$ : Heat capacity of water in $$J\,kg^{-1}\,K^{-1}$$      

# Balance Equation

The first step in a simulation is the formation of a model equation. Thus, you should dig up your old lecture notes on the subject of heat and mass transfer and construct the heat balance equation. We will assume an ideal mixing inside the tank, so that the outlet temperature is identical to the temperature at any point inside the tank.

The energy balance states that the changes of energy inside the tank is equal to the amount of energy that goes in and out of the system.

\begin{equation} \label{eq:heatBalance}
    \frac{dU}{dt} = H_{\alpha} - H_{\omega} + Q
\end{equation}

The change in internal energy can be represented as $$\frac{dU}{dt} = V\,\rho\,c_v\,\frac{dT_R}{dt}$$. Further, we assume that water is incompressible, hence $$c_v = c_p$$. The enthalpy stream can be described with $$\dot{H}_i = F\,\rho\,c_p\,T_i$$, With the assumption of ideal mixing, the outflow temperature $$T_\omega$$ is equal to the temperature of the tank $$T_R$$. The result is the following first-order ordinary differential equation:

\begin{align} \label{eq:tempOde}
    \frac{d T_R}{dt} = \frac{F}{V}(T_{\alpha} - T_R) + \frac{Q}{V\,\rho\,c_p}
\end{align}

The balance equation gives us an initial-value problem with $$T_R(t=0) = T_\alpha$$.

# Numerical Methods: Runge-Kutta-Method

Despite the fact that the above equation can be analitically solved for a constant heat flow rate $$Q$$, we will use a numerical method to solve the differential equation, the classical Runge-Kutta-Method. Suppose we have the value of $$T_{R,i} \equiv y_i$$ at a specific time $$t_i$$, the value of $$y(t_{i+1})$$ for the time $$t_{i+1} = t_i + \Delta{}t$$ can be approximated with the weighted sums of the differential itself. With $$f(t, y) = \frac{dT_R}{dt}$$:

| | | | |
|-|-|-|-|
| $$t_1 = t_i$$ | $$y_1 = y_i$$ | $$p_1 = f(t_1, y_1)$$ | $$k_1 = \Delta{}t\,p_1$$ |
| $$t_2 = t_i + \frac{\Delta{}t}{2}$$ | $$y_1 = y_i + \frac{k_1}{2}$$ | $$p_2 = f(t_2, y_2)$$ | $$k_2 = \Delta{}t\,p_2$$ |
| $$t_3 = t_i + \frac{\Delta{}t}{2}$$ | $$y_3 = y_i + \frac{k_2}{2}$$ | $$p_3 = f(t_3, y_3)$$ | $$k_3 = \Delta{}t\,p_3$$ |
| $$t_4 = t_i + \Delta{}t$$ | $$y_4 = y_i + k_3$$ | $$p_4 = f(t_4, y_4)$$ | $$k_4 = \Delta{}t\,p_4$$ |

Consequently, the value of $$y(t_{i+1})$$ will be
\begin{align} \label{eq:RungeKutta}
    y(t_{i+1}) = y_i + \frac{k_1 + 2k_2 + 2k_3 + k_4}{6}
\end{align}


# Implementation with Python

It is time to implement the simulation in python. We will also use the numpy library for mathematical computing and matplotlib library for plotting. First, we import these libraries and define the differential equation as a function.


```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def temperature_ode(t: float, T_R: float, params: dict):
    '''Differential equation of the heat balance'''
    T_in = params['T_in']
    F = params['F']
    V = params['V']
    rho = params['rho']
    c_p = params['c_p']
    Q = params['Q']
    
    return (F/V*(T_in - T_R)) + (Q/(V*rho*c_p))
```

Next, we define the parameters as python variables. You might have noticed the `params` argument in the `temperature_ode` function. All of the following parameters will be then passed to the function as one dictionary, a data-type in python. I find this method to be quite convenient.


```python
F = 0.15e-3    # m^3 s^-1
V = 0.01       # m^3
T_in = 293.15  # K
T_R = T_in     # K
Q = 600        # W; initial heat flow rate
rho = 997      # kg m^-3
c_p = 4.186    # J kg^-1
T_set = 315.15 # K

dt = 0.1       # s, time step for numerically solving the differential equation

params = {'F': F, 'V': V, 'T_in': T_in, 'T_R': T_R, 'Q': Q, 'rho': rho,
          'c_p': c_p, 'T_set': T_set, 'dt': dt}
```

In the following function, we define one step of a Runge-Cutta computation.


```python
def runge_kutta_step(fun: callable, t_0: float, y_0: float, dt: float, *args):
    '''Perform one step of classical Runge-Kutta Solver for initial-value problem'''
    
    t_1 = t_0
    y_1 = y_0
    k_1 = dt * fun(t_1, y_1, *args)

    t_2 = t_1 + 0.5*dt
    y_2 = y_1 + 0.5*k_1
    k_2 = dt * fun(t_2, y_2, *args)

    t_3 = t_2
    y_3 = y_1 + 0.5*k_2
    k_3 = dt * fun(t_3, y_3, *args)

    t_4 = t_1 + dt
    y_4 = y_1 + k_3
    k_4 = dt * fun(t_4, y_4, *args)

    t = t_0 + dt
    k = k_1/6 + k_2/3 + k_3/3 + k_4/6
    y = y_0 + k
    
    return (t, y)
```

And now all is set. We can start the simulation. As the temperature output of the tank should be $$T_\omega = 42^\circ{}C$$, we have to calculate the specific heat flow rate needed to reach this end temperature. For this purpose, we will set the balance equation to its stationary state, meaning every time-derivative (in this case $$\frac{dT_R}{dt}$$) is set to zero. The resulting equation can be solved to the required heat flow rate

\begin{equation}\label{eq:stationaryQ}
    Q = F\,\rho\,c_p\,(T_R - T_{\alpha})
\end{equation}


```python
Q_stat = F*rho*c_p*(T_set - T_in)
params['Q'] = Q_stat

print(f'For an output temperature of {T_set:.2f} K,' \
      f' the required heat flow rate is {Q_stat:.2f} W.')
```

    For an output temperature of 315.15 K, the required heat flow rate is 13.77 W.


Thus we will use this heat flow rate in the simulation. To give an idea, how $$T_R$$ varies with $$Q$$, we will also simulate the temperature for different $$Q$$s.


```python
params_1 = params.copy()
params_1['Q'] = Q_stat / 2
params_2 = params.copy()
params_3 = params.copy()
params_3['Q'] = Q_stat * 2

t_start = 0
t_end = 400

time = np.arange(t_start, t_end+dt*0.1, dt)
temp_1 = np.zeros(len(time))
temp_2 = np.zeros(len(time))
temp_3 = np.zeros(len(time))

for idx, t in enumerate(time):
    if idx == 0:
        # Set the initial values
        temp_1[idx] = T_R
        temp_2[idx] = T_R
        temp_3[idx] = T_R
    else:
        _, temp_1[idx] = runge_kutta_step(temperature_ode, t, 
                                          temp_1[idx-1], dt, params_1)
        _, temp_2[idx] = runge_kutta_step(temperature_ode, t, 
                                          temp_2[idx-1], dt, params_2)
        _, temp_3[idx] = runge_kutta_step(temperature_ode, t, 
                                          temp_3[idx-1], dt, params_3)

# Plot result
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(time, temp_1-273.15, label=f'$Q = {Q_stat/2:.2f}\,W$')
ax.plot(time, temp_2-273.15, '-g', label=f'$Q = {Q_stat:.2f}\,W$')
ax.plot(time, temp_3-273.15, label=f'$Q = {Q_stat*2:.2f}\,W$')
ax.plot([min(time), max(time)], [42, 42], '--')
ax.grid('on')
ax.legend()
ax.set_xlabel('Time $t$ in s')
ax.set_ylabel('Temperature $T_R$ in $^\circ{}C$')
ax.set_xlim([time[0], time[-1]])
fig.tight_layout()
```


![png](/assets/images/2020-07-02-Temperature-Control-of-Water-Heater/output_12_0.png)


And the simulation is done. As you can see, our water heater would need roughly $350\,s$ under the given conditions before reaching the desired temperature. You probably wouldn't want to wait so long (especially in the morning) to get the pleasant warm water. And that is where the concept of control comes in.

# Temperature Control

In the previous figure, we have seen that the stationary outlet temperature $$T_R$$ depends on the heat input $$Q$$, where a higher heat rate leads to a higher temperature. If we look at the first 50 seconds, we also see that the higher heat rate results in a faster rise in outlet temperature. Therefore, one idea of control is to change an actuator (in this case the heat flow rate) over the course of time to reach the desired output as quickly as possible.

For our case, we will use a feedback controller with classical tuning rules. As a first step, we will take the balance equation:

\begin{align}
    \frac{d T_R}{dt} = \frac{F}{V}(T_{\alpha} - T_R) + \frac{Q}{V\,\rho\,c_p}
\end{align}

and perform a Laplace-Transformation on it. Suppose that the laplace transformation of the controlled variable is $$\mathcal{L}\{T_R\} = Y(s)$$ and the control signal is $$\mathcal{L}\{Q\} = U(s)$$:

\begin{gather}
    s\,Y(s) - T_R(t=0) = \frac{F}{V}\left(s\,T_{\alpha} - Y(s)\right) + \frac{1}{V\,\rho\,c_p}U(s)
\end{gather}
\begin{gather}
    \Leftrightarrow\quad \left(s + \frac{F}{V}\right)\,Y(s) = \frac{1}{V\,\rho\,c_p}U(s) + \frac{s\,F\,T_\alpha}{V} + T_R(t=0)
\end{gather}
\begin{gather}
    \Leftrightarrow\quad Y(s) = \frac{1}{V\,\rho\,c_p\,\left(s + \frac{F}{V}\right)}U(s) + \frac{s\,F\,T_\alpha}{V\,\left(s + \frac{F}{V}\right)} + \frac{T_R(t=0)}{\left(s + \frac{F}{V}\right)}
\end{gather}

In this frequency domain, the system is defined as the transfer function of the control signal $U(s)$ to the controlled variable $Y(s)$.

\begin{align}
    G(s) &= \frac{1}{V\,\rho\,c_p\,\left(s + \frac{F}{V}\right)} = \frac{1}{V\,\rho\,c_p\,s + F\,\rho\,c_p} \nonumber \\
\end{align}
\begin{align}
    \Rightarrow G(s) &= \frac{\frac{1}{F\,\rho\,c_p}}{\frac{V}{F}\,s + 1} \equiv \frac{K}{T_1\,s + 1}
\end{align}

In the classical tuning rules by **Ziegler** and **Nichols**, a PI-Controller for the so called PT$$_1$$-Model above is obtained via the rule $$K_P = \frac{1}{K}$$ and $$T_I = 0.07\,T_1$$.


```python
# Define controller parameter
K = 1 / (F * c_p * rho)
T_1 = V / F

K_P = 1 / K
T_I = T_1 * 0.7

print(f"The obtained controller parameters are: K_P = {K_P:.2f}, T_I = {T_I:.2f}")
```

    The obtained controller parameters are: K_P = 0.63, T_I = 46.67


We can now simulate these parameters together with the system differential equation from above. To do this, we would also need the following PI-controller equation

\begin{align}
    Q(t) = K_P\,e(t) + \frac{K_P}{T_I}\int_0^t e(\tau)\,d\tau
\end{align}

The argument $$e(t)$$ in this integral refers to the error of the controlled variable to its reference signal $$e(t) = T_{set} - T_R(t)$$. In the implementation, we would approximate the integral with the sum.


```python
def pi_controller(epsilon: np.ndarray, K_P, T_I, dt):
    '''
    Implementation of PI-Controller
    
    epsilon is the error of the controlled variable to its reference, saved as a numpy array.
    '''
    return K_P * epsilon[-1] + (K_P/T_I)*np.sum(np.multiply(epsilon, dt))
```

All is set, we can now begin the simulation.


```python
t_start = 0
t_end = 400

time = np.arange(t_start, t_end+dt*0.1, dt)
temp = np.zeros(len(time))
Q = np.zeros(len(time))
epsilon = np.array([])

for idx, t in enumerate(time):
    if idx == 0:
        # Set the initial values
        temp[idx] = T_R
    else:
        epsilon = np.append(epsilon, (T_set - temp[idx-1]))
        Q[idx] = pi_controller(epsilon, K_P, T_I, dt)
        params['Q'] = Q[idx]
        _, temp[idx] = runge_kutta_step(temperature_ode, t, temp[idx-1], dt, params)

# Plot result
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(time, temp-273.15, 'g')
ax.plot([min(time), max(time)], [42, 42], '--')
ax.grid('on')
ax.set_xlabel('Time $t$ in s')
ax.set_ylabel('Temperature $T_R$ in K')
ax.set_xlim([time[0], time[-1]])
ax.yaxis.label.set_color('g')
ax.tick_params(axis='y', colors='g')

ax2 = ax.twinx()
ax2.plot(time, Q, 'r')
ax2.set_ylabel('Heat flow rate $Q$ in W')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')

fig.tight_layout()
```


![png](/assets/images/2020-07-02-Temperature-Control-of-Water-Heater/output_19_0.png)


As you can see, the temperature of the water flowing out of the heater would now reach the desired temperature within roughly 150 s, instead of 300 s as shown in the previous figure. The controller would be able to reduce your waiting time by 50%! On the downside, notice that there are ranges where the water temperature would have exceeded the $$42 ^\circ C$$ line. While you probably wouldn't mind a temperature a little bit higher than what you would have set, such oscillating behaviour is quite often undesirable, if not harmful in some occasions. In any cases, an iteration or two of controller's optimization should be done and controllers from a quick tuning rule are far from optimized. Regardless, you should have gotten an idea, how such control system works in general.

### The end

You can find this notebook in this [link](https://github.com/FelixFebrian/FelixFebrian.github.io/blob/master/notebooks/2020-07-02-Temperature-Control-of-Water-Heater.ipynb). Let me know that you like (or don't like) this article and any topic ideas are warmly welcomed.


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: [
      "MathMenu.js",
      "MathZoom.js",
      "AssistiveMML.js",
      "a11y/accessibility-menu.js"
    ],
    jax: ["input/TeX", "output/CommonHTML"],
    TeX: {
      extensions: [
        "AMSmath.js",
        "AMSsymbols.js",
        "noErrors.js",
        "noUndefined.js",
      ]
    }
  });
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>