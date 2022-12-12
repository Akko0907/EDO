import numpy as np
import matplotlib.pyplot as plt
from Solve import EDO

def Osc_harm():
    omega = 4.
    F_list = [lambda y1,y2: -omega**2*y2, lambda y1,y2: y1]
    h = 0.05
    T = 4
    # Initial conditions
    v0 = 0.
    x0 = 0.2
    init = np.array([v0,x0])

    N = int(T//h)

    return omega, F_list, h, T, init

def Epidem():
    N = 150
    T = 150
    h = T/N
    beta = 1/2
    gama = 1/3
    F_list = [lambda y1,y2,y3: -beta*y1*y2, lambda y1,y2,y3: beta*y1*y2-gama*y2, lambda y1,y2,y3: gama*y2 ]

    # Initial conditions
    s = 1
    j = 1.27e-6
    r = 0
    init = np.array([s,j,r])

    return F_list,h,T,init

def Terminal_V():
    N = 50
    T = 20
    h = T/N
    B = 0.01
    g = 9.8
    F_list = [lambda y1,y2: g-B*y1**2/2, lambda y1,y2: y1]

    # Initial conditions
    x0 = 100.
    v0 = 0.
    init = np.array([v0,x0])

    return F_list,h,T,init

def Prob1():
    F_list = [lambda y,t: -t*y, lambda y,t: 1]
    h = 0.05
    T = 1
    # Initial conditions
    y0 = 1.
    t0 = 0.
    init = np.array([y0,t0])

    N = int(T//h)

    return F_list,h,T,init


omega,F_list,h,T,init = Osc_harm()
oscilador_harmonico = EDO(F_list,2,0.1)

t,vs,xs = oscilador_harmonico(init,'special')
t,vb,xb = oscilador_harmonico(init,'base')
t,vm,xm = oscilador_harmonico(init,'mod')
t,vr2,xr2 = oscilador_harmonico(init,'rk2')
t,vr4,xr4 = oscilador_harmonico(init,'rk4')

plt.plot(t,xs,label="Special")
plt.plot(t,xb,label="Euler")
plt.plot(t,xm,label="Modified Euler")
plt.plot(t,xr,label="Runge Kutta 2")
plt.plot(t,xr4,label="Runge Kutta 4")
#plt.scatter(t,xs)
#plt.scatter(t,xb)
#plt.scatter(t,xm)
#plt.scatter(t,xr)
#plt.scatter(t,xr4)
plt.plot(t,0.2*np.cos(omega*t),label="Exact Solution")
plt.legend()
plt.show()
