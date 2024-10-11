import os
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference.models.utils import get_roboflow_model

load_dotenv()

#SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
SOURCE = np.array([[387, 287], [1454, 260], [1908, 537], [-82, 595]])  # array of coordinates of field plane
TARGET_WIDTH = 16.59128
TARGET_HEIGHT = 8.211312

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)  # array of coordinates of the target plane

model = get_roboflow_model(model_id="frc-scouting-application/2", api_key=os.getenv('ROBOFLOW_API_KEY'))
video_info = sv.VideoInfo.from_video_path(video_path="video.mp4")
frames_generator = sv.get_video_frames_generator(source_path='video.mp4')
tracker = sv.ByteTrack(frame_rate=video_info.fps)  # initiates tracker

thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(
    resolution_wh=video_info.resolution_wh)
# calculates optimal text scale and thickness for labels and other annotators

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    color_lookup=sv.ColorLookup.TRACK,
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps,
    color_lookup=sv.ColorLookup.TRACK,
)
# initiates all annotators

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


polygon_zone = sv.PolygonZone(SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))  # keeps dictionary of historical point movement

with sv.VideoSink(target_path="result.mp4", video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[polygon_zone.trigger(detections)]  # restricts detections to polygon zone
        detections = tracker.update_with_detections(detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        labels = []  # sets an empty label array
        for tracker_id in detections.tracker_id:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time
            labels.append(f"#{tracker_id} {int(speed)}")

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        sink.write_frame(frame=annotated_frame)
