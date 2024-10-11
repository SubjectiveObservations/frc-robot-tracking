import supervision as sv
from inference.models.utils import get_roboflow_model

import os
load_dotenv()
model = get_roboflow_model(model_id="frc-scouting-application/2", api_key=os.getenv('ROBOFLOW_API_KEY'))
heat_map_annotator = sv.HeatMapAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path='video.mp4')
frames_generator = sv.get_video_frames_generator(source_path='video.mp4')

with sv.VideoSink(target_path="resultp.mp4", video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = heat_map_annotator.annotate(
            scene=frame.copy(),
            detections=detections)
        sink.write_frame(frame=annotated_frame)