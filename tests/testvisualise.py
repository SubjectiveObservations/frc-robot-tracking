import cv2
import matplotlib.pyplot as plt
import numpy as np

# Your source and target points
source = np.array([[387, 287], [1454, 260], [1908, 537], [-82, 595]], dtype=np.float32)
TARGET_WIDTH = 16.59128
TARGET_HEIGHT = 8.211312
target = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
], dtype=np.float32)

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

# Example points to transform
points = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

# Transform points
transformed_points = transformer.transform_points(points)

# Create an image to draw points on
image = np.zeros((600, 600, 3), dtype=np.uint8)

# Draw original points in blue
for point in points:
    cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

# Draw transformed points in red
for point in transformed_points:
    cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
