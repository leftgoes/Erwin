import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable

T = float | np.ndarray

# h_bar = 1, L = 1
class FiniteDiff:
    def __init__(self, n_x: int, n_t: int, dt: float = 1e-9) -> None:
        self.n_x = n_x
        self.n_t = n_t
        self.dx = 1/(n_x - 1)
        self.dt = dt
        self.psi: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0, 1, self.n_x)

    @staticmethod
    def V_gaussian(a: float = 1e4, sigma: float = 1/10, mu: float = 1/2) -> Callable[[T], T]:
        return lambda x: -a * np.exp(-(x - mu)**2/(2 * sigma**2))

    @staticmethod
    def psi0_sin(n: int = 1) -> Callable[[T], T]:
        return lambda x: 2**(1/2) * np.sin(n * np.pi * x)

    @staticmethod
    def normalize(psi: np.ndarray):
        return psi / np.sum(np.abs(psi))

    def show_potential(self):
        plt.plot(self.x, self.V_gaussian()(self.x))
        plt.show()
        plt.close()

    def fdm(self, psi0: Callable[[T], T] | None = None, V: Callable[[T], T] | None = None):
        print(self.dt/self.dx**2)
        x = self.x
        if V is None: V = self.V_gaussian(a=1e4, sigma=1/10, mu=0.5)
        if psi0 is None: psi0 = self.psi0_sin(n=1)
        potential = V(x)

        self.psi = np.zeros((self.n_t, self.n_x), dtype=np.complex_)
        self.psi[0, :] = self.normalize(psi0(x))
    
        for m, psi_m in enumerate(self.psi[:-1]):
            print(f'\r{round(100 * m/self.n_t, 2)}%', end='')
            for j, psi in enumerate(psi_m):
                psi_left = 0 if j == 0 else self.psi[m, j - 1]
                psi_right = 0 if j == self.n_x - 1 else self.psi[m, j + 1]
                self.psi[m + 1, j] = psi + 1j/2 * self.dt/self.dx**2 * (psi_left - 2 * psi + psi_right) - 1j * potential[j] * self.dt * psi
            self.psi[m + 1] = self.normalize(self.psi[m + 1])

    def animate(self, frames: int = 3000, fps: float = 30):
        x = self.x
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(min(self.psi.real.min(), np.abs(self.psi).min()), max(self.psi.real.max(), np.abs(self.psi).max()))
        line_abs, = plt.plot([], [], 'r-')
        line_real, = plt.plot([], [], 'b--')

        def frame(m: int) -> None:
            print(f'\r{m}', end='')
            line_abs.set_data(x, np.abs(self.psi[m]))
            line_real.set_data(x, self.psi[m].real)
        animation = FuncAnimation(fig, func=frame, frames=range(0, self.n_t, round(self.n_t/frames)), interval=1000/fps)
        animation.save('video.mp4', fps=fps, dpi=300)


if __name__ == '__main__':
    finite = FiniteDiff(X_SAMPLES, T_SAMPLES, DELTA_T)
    finite.show_potential()
    finite.fdm()
    finite.animate(FRAMES_COUNT)
