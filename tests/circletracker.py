import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


class PointTracker:
    def __init__(self, initial_position):
        self.positions = [initial_position]

    def update_position(self, new_position):
        self.positions.append(new_position)

    def save_path_as_gif(self, filename='point_path.gif', fps=10):
        fig, ax = plt.subplots()
        ax.set_xlim(min(p[0] for p in self.positions) - 1, max(p[0] for p in self.positions) + 1)
        ax.set_ylim(min(p[1] for p in self.positions) - 1, max(p[1] for p in self.positions) + 1)
        point, = ax.plot([], [], 'bo')
        path_line, = ax.plot([], [], 'r--', linewidth=0.5)

        def init():
            point.set_data([], [])
            path_line.set_data([], [])
            return point, path_line

        def update(frame):
            x, y = self.positions[frame]
            point.set_data([x], [y])
            path_line.set_data([p[0] for p in self.positions[:frame+1]], [p[1] for p in self.positions[:frame+1]])
            return point, path_line

        ani = FuncAnimation(fig, update, frames=len(self.positions), init_func=init, blit=True)
        ani.save(filename, writer=PillowWriter(fps=fps))

# Example usage
tracker = PointTracker((0, 0))

# Simulate some movements
for t in np.linspace(0, 2*np.pi, 100):
    new_position = (np.cos(t), np.sin(t))  # Circular path example
    tracker.update_position(new_position)

# Save the path as a GIF
tracker.save_path_as_gif()

plt.show()
