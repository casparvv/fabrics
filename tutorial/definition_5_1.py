import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

n = 2
q = ca.SX.sym("q", n)
q_d = ca.SX.sym("q_d", n)
qdot = ca.SX.sym("qdot", n)

r = ca.norm_2(q)
a = ca.arctan2(q[1], q[0])
w = ca.norm_2(qdot)/ca.norm_2(q)
h = ca.SX.sym("h", 2)
b = r
h[0] = b * r * w**2 * ca.cos(a)
h[1] = b * r * w**2 * ca.sin(a)
h_fun = ca.Function("h", [q, qdot], [h])


class Accelerator(object):
    def __init__(self):
        self._gamma = 1.0

    def update(self, t):
        self._gamma = np.sin(t)

    def gamma(self, qdot):
        return self._gamma * qdot

class Geometry(object):

    """Geometry as in Optimization fabrics
        xddot + h(x, xdot) = 0
    """

    def __init__(self):
        self._n = 2
        self._h = np.zeros(n)
        self._rhs = np.zeros(n)
        self._rhs_aug = np.zeros(2*n)
        self._q = np.zeros(n)
        self._qdot = np.zeros(n)

    def setRHS(self):
        self._h = h_fun(self._q, self._qdot)
        self._rhs = -self._h

    def addAccelerator(self, Acc, t):
        Acc.update(t)
        self._rhs += Acc.gamma(self._qdot)

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def contDynamics(self, z, t, Acc=None):
        self._q = z[0:n]
        self._qdot = z[n:2*n]
        self.setRHS()
        if Acc:
            self.addAccelerator(Acc, t)
        self.augment()
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, t, Acc=None):
        sol, info = odeint(self.contDynamics, z0, t, args=(Acc,), full_output=True)
        return sol

def update(num, x1, x2, y1, y2, line1, line2, point1, point2):
    start = max(0, num - 100)
    line1.set_data(x1[start:num], y1[start:num])
    point1.set_data(x1[num], y1[num])
    line2.set_data(x2[start:num], y2[start:num])
    point2.set_data(x2[num], y2[num])
    return line1, point1, line2, point2

def plotTraj(sol, ax, fig):
    x = sol[:, 0]
    y = sol[:, 1]
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.plot(x, y)
    (line,) = ax.plot(x, y, color="k")
    (point,) = ax.plot(x, y, "rx")
    return (x, y, line, point)

def main():
    # setup 
    geo = Geometry()
    acc = Accelerator()
    w0 = 1.0
    r0 = 2.0
    a0 = 1.0/3.0 * np.pi
    q0 = r0 * np.array([np.cos(a0), np.sin(a0)])
    q0_dot = np.array([-r0 * w0 * np.sin(a0), r0 * w0 * np.cos(a0)])
    t = np.arange(0.0, 20.00, 0.01)
    z0 = np.concatenate((q0, q0_dot))
    # solving
    sol = geo.computePath(z0, t)
    sol_en = geo.computePath(z0, t, Acc=acc)
    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Geometry generators")
    ax[0].set_title("Remaining on the same energy level")
    ax[1].set_title("Changing energy level with $\gamma = sin(t) \dot{q}$")
    (x, y, line, point) = plotTraj(sol, ax[0], fig)
    (x2, y2, line2, point2) = plotTraj(sol_en, ax[1], fig)
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, x2, y, y2, line, line2, point, point2],
        interval=25, blit=True
    )
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()
