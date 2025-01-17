import os

import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference.models.utils import get_roboflow_model

load_dotenv()

model = get_roboflow_model(model_id="frc-scouting-application/2", api_key=os.getenv('ROBOFLOW_API_KEY'))
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path="video.mp4")
frames_generator = sv.get_video_frames_generator(source_path='video.mp4')

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    
    labels = [
        f"#{tracker_id} {class_id}"
        for class_id, tracker_id
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