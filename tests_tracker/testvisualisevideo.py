import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture('people-walking.mp4')

# Define the source and target points for perspective transformation
source = np.array([[10, 10], [200, 10], [200, 200], [10, 200]], dtype=np.float32)
target = np.array([[30, 30], [180, 50], [190, 190], [50, 180]], dtype=np.float32)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Initialize the ViewTransformer
transformer = ViewTransformer(source, target)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Example points to transform
    points = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

    # Transform points
    transformed_points = transformer.transform_points(points)

    # Draw original points in blue
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    # Draw transformed points in red
    for point in transformed_points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
