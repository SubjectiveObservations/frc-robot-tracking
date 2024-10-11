import os

import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference.models.utils import get_roboflow_model

load_dotenv()

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

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    
    labels = [
        f"#{tracker_id}"
        for tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=detections)

    with sv.CSVSink("tracking_results2.csv") as sink:
        #for frame_index, frame in enumerate(frames_generator):
        sink.append(detections, {})

sv.process_video(
    source_path="video.mp4",
    target_path="resultc2.mp4",
    callback=callback
)