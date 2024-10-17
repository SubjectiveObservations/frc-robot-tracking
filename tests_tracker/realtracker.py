from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


class PointTracker:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def save_path_as_gif(self, filename='points_path.gif', fps=10):
        fig, ax = plt.subplots()
        all_x = [x for points in self.coordinates.values() for x, y in points]
        all_y = [y for points in self.coordinates.values() for x, y in points]
        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        points_plot, = ax.plot([], [], 'bo')
        path_lines = {key: ax.plot([], [], 'r--', linewidth=0.5)[0] for key in self.coordinates.keys()}

        def init():
            points_plot.set_data([], [])
            for path_line in path_lines.values():
                path_line.set_data([], [])
            return points_plot, *path_lines.values()

        def update(frame):
            all_points = []
            for key, points in self.coordinates.items():
                if len(points) > frame:
                    x, y = points[frame]
                    all_points.append((x, y))
                    path_lines[key].set_data([p[0] for p in list(points)[:frame+1]], [p[1] for p in list(points)[:frame+1]])
            if all_points:
                points_plot.set_data([p[0] for p in all_points], [p[1] for p in all_points])
            return points_plot, *path_lines.values()

        ani = FuncAnimation(fig, update, frames=max(len(points) for points in self.coordinates.values()), init_func=init, blit=True)
        ani.save(filename, writer=PillowWriter(fps=fps))

# Example usage
coordinates = defaultdict(lambda: deque(maxlen=100))  # Replace with video_info.fps
# Simulate some movements for two points
for t in range(100):
    coordinates['point1'].append((np.cos(t * 0.1), np.sin(t * 0.1)))
    coordinates['point2'].append((np.cos(t * 0.1 + np.pi / 4), np.sin(t * 0.1 + np.pi / 4)))

tracker = PointTracker(coordinates)
tracker.save_path_as_gif()

plt.show()
