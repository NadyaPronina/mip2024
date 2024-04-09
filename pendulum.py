import pybullet as p
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def cost(traj):
    sz = len(traj)
    l2 = np.sqrt(np.sum(traj ** 2 / sz))
    linf = np.max(np.abs(traj))
    return (l2, linf)

def symplectic_euler(func, x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        q_prev, p_prev = x[i - 1]
        p_next = p_prev + h * func([q_prev, p_prev], 0)[1]
        q_next = q_prev + h * p_next
        x[i] = [q_next, p_next]
    return x

def symplectic_euler_param(func, x0, t, a, r, m, tau):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        q_prev, p_prev = x[i - 1]
        p_next = p_prev + h * func([q_prev, p_prev], 0, a, r, m, tau)[1]
        q_next = q_prev + h * p_next
        x[i] = [q_next, p_next]
    return x

g = 10
L = 0.5
dt = 1/240 # pybullet simulation step
q0 = np.pi - np.deg2rad(15)   # starting position (radian)
jIdx = 1
MaxTime = 0.1
logTime = np.arange(0.0, MaxTime, dt)
sz = len(logTime)
logPos = np.zeros(sz)
logPos[0] = q0
idx = 0

physicsClient = p.connect(p.DIRECT) # or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)

# turn off internal damping
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetPosition=q0, controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
for t in logTime[1:]:
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, controlMode=p.TORQUE_CONTROL, force=0.1)
    p.stepSimulation()

    jointState = p.getJointState(boxId, jIdx)
    th1 = jointState[0]
    idx += 1
    logPos[idx] = th1


def rp(x, t):
    r = 0.5 
    m = 2
    tau = 0.1
    return [x[1], 
            -g/L * np.sin(x[0]) -  (r * x[1])/(m * L**2) + tau / (m * L**2)] 

def rp_lin(x, t):
    r = 0.5 
    m = 2
    tau = 0.1
    return [x[1], 
            g/L * (x[0] - np.pi) -  (r * x[1])/(m * L**2) + tau / (m * L**2)]  


def rp_param(x, t, a, r, m, tau):
    return [x[1], 
            -a*np.sin(x[0]) -  (r * x[1])/(m * L**2) + tau / (m * L**2)]

theta = odeint(rp, [q0, 0], logTime)
logTheor = theta[:,0]

(l2, linf) = cost(logPos - logTheor)
print(l2, linf)

theta = symplectic_euler(rp, [q0, 0], logTime)
logEuler = theta[:, 0]

(l2, linf) = cost(logPos - logEuler)
print(l2, linf)

thetaLin = symplectic_euler(rp_lin, [q0, 0], logTime)
logLin = thetaLin[:, 0]

def l2_cost(params):
    a, r, m, tau = params
    xx = symplectic_euler_param(rp_param, [q0, 0], logTime, a, r, m, tau)
    return cost(logPos-xx[:,0])[0]

a0 = [0.9*g/L]
r0 = 0.5
m0 = 2
tau0 = 0.1
my_params = np.array([a0[0], r0, m0, tau0])
print(f'init a: {a0}')
res = minimize(l2_cost, my_params)
print(f'l2_res: {res.fun}')
print(f'a: {res.x}')
print(f'a_th: {g/L}')
# a = res.x[0]

import matplotlib.pyplot as plt

plt.grid(True)
plt.plot(logTime, logPos, label = "simPos")
plt.plot(logTime, logTheor, label = "simTheor")
plt.plot(logTime, logEuler, label = "simEuler")
plt.plot(logTime, logLin, label = "logLin")
plt.legend()

plt.show()

p.disconnect()

# 1 add damping to rp and rp_lin
# 2 add control torque to rp and rp_lin
# 3 widen minimize to estimate a, damping coef and toqrue coef
# 4 derive linear model from upper position and compare to sim (compare first 0.1 sec)
