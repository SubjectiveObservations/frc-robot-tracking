import os
from collections import defaultdict, deque

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference.models.utils import get_roboflow_model
from matplotlib.animation import FuncAnimation, PillowWriter

load_dotenv()

#SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
SOURCE = np.array([[387, 287], [1454, 260], [1908, 537], [-82, 595]])  # array of coordinates of field plane
TARGET_WIDTH = 16591.28
TARGET_HEIGHT = 8211.312

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)  # array of coordinates of the target plane

model = get_roboflow_model(model_id="frc-scouting-application/3", api_key=os.getenv('ROBOFLOW_API_KEY'))
video_info = sv.VideoInfo.from_video_path(video_path="video.mp4")
frames_generator = sv.get_video_frames_generator(source_path='video.mp4')
tracker = sv.ByteTrack(frame_rate=video_info.fps)  # initiates tracker

class ViewTransformer:
    # class to pass all coordinates of SOURCE to opencv2 to transform it into a target plane
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)  # ensures np arrays are in float32 format for opencv library
        self.m = cv2.getPerspectiveTransform(source, target)

#Takes two arrays of points, source and target, and calculates the perspective transformation matrix self.m using OpenCV’s cv2.getPerspectiveTransform function.

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
# Takes an array of points, checks if the array is empty, and then applies the perspective transformation using cv2.perspectiveTransform function, transforming the points from the source perspective to the target perspective. The transformed points are then reshaped back into the original format.
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)  # removes third dimension

class PointTracker:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.colors = {key: (random.random(), random.random(), random.random()) for key in coordinates.keys()}

    def save_path_as_gif(self, filename='tracking_path4.gif', fps=30):
        fig, ax = plt.subplots()
        all_x = [x for points in self.coordinates.values() for x, y in points]
        all_y = [y for points in self.coordinates.values() for x, y in points]

        if all_x and all_y:
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        points_plot, = ax.plot([], [], 'bo')
        path_lines = {key: ax.plot([], [], linestyle='--', linewidth=0.5, color=self.colors[key])[0] for key in self.coordinates.keys()}

        def init():
            points_plot.set_data([], [])
            for path_line in path_lines.values():
                path_line.set_data([], [])
            return points_plot, *path_lines.values()

        def update(frame):
            all_points = []
            for key, points in self.coordinates.items():
                if points:
                    points_up_to_frame = [points[i] for i in range(frame + 1) if i < len(points)]
                    if points_up_to_frame:
                        x, y = points[frame]
                        all_points.append((x, y))
                        path_lines[key].set_data([p[0] for p in points_up_to_frame], [p[1] for p in points_up_to_frame])
            if all_points:
                points_plot.set_data([p[0] for p in all_points], [p[1] for p in all_points])
            return points_plot, *path_lines.values()

        # Ensure frames_count calculation avoids empty sequences
        frames_count = max((len(points) for points in self.coordinates.values() if points), default=0)
        if frames_count == 0:
            print("No data to plot")
            return

        ani = FuncAnimation(fig, update, frames=frames_count, init_func=init, blit=True)
        ani.save(filename, writer=PillowWriter(fps=fps))

polygon_zone = sv.PolygonZone(SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))  # keeps dictionary of historical point movement

with sv.CSVSink("tracked6.csv") as sink:
    for frame_index, frame in enumerate(frames_generator):
        speed = "N/A"
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[polygon_zone.trigger(detections)]  # restricts detections to polygon zone  # associates trackers
        detections = tracker.update_with_detections(detections)

        # associates trackers
        detections = tracker.update_with_detections(detections)
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        
        for tracker_id, point in zip(detections.tracker_id, points):
            point_tuple = tuple(point)  # convert numpy.ndarray to tuple
            coordinates[tracker_id].append(point_tuple)

        # labels = []  # sets an empty label array
        # for tracker_id in detections.tracker_id:
        #     if len(coordinates[tracker_id]) < video_info.fps / 2:
        #         speed = "N/A"
        #         labels.append(f"#{tracker_id}")
        #     else:
        #         coordinate_start = coordinates[tracker_id][-1]
        #         coordinate_end = coordinates[tracker_id][0]
        #         distance = abs(coordinate_start - coordinate_end)
        #         time = len(coordinates[tracker_id]) / video_info.fps
        #         speed = np.linalg.norm(distance) / (time * 1000)
        #         labels.append(f"#{tracker_id} {float(speed)} ms^-1")
            
        # sink.append(detections, {
        #     "frame_index": frame_index,
        #     # "speed": speed,
        #     # "coordinate": coordinates
        # })

tracker = PointTracker(coordinates)
tracker.save_path_as_gif(filename='tracking_path4.gif', fps=video_info.fps)
