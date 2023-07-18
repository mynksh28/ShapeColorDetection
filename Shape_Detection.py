import cv2
import numpy as np

# Area threshold
MIN_SHAPE_AREA = 1200

# Detect shapes
def detect_shapes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours of the detected edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= MIN_SHAPE_AREA:
            # Approximate the contour to reduce the number of vertices
            approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
            num_vertices = len(approx)

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            shape = {
                "contour": contour,
                "num_vertices": num_vertices,
                "center": (center_x, center_y),
                "area": area,
                "circularity": circularity,
                "coordinates": (center_x, center_y)
            }

            detected_shapes.append(shape)

    return detected_shapes


def identify_shapes(detected_shapes, image):
    identified_shapes = []

    for shape in detected_shapes:
        num_vertices = shape["num_vertices"]
        circularity = shape["circularity"]

        label = "Unknown"
        color = "Unknown"

        # Identify the shape based on the number of vertices
        if num_vertices == 3:
            label = "Triangle"
        elif num_vertices == 4:
            x, y, width, height = cv2.boundingRect(shape["contour"])
            aspect_ratio = float(width) / height
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                label = "Square"
            else:
                label = "Rectangle"
        elif num_vertices == 5:
            label = "Pentagon"
        elif num_vertices == 6:
            label = "Hexagon"
        elif circularity >= 0.85:
            label = "Circle"

        center_x, center_y = shape["center"]
        pixel_color = image[center_y, center_x]
        b, g, r = pixel_color

        # Identify the color based on RGB values
        if r > 200 and g < 50 and b < 50:
            color = "Red"
        elif r > 200 and g < 50 and b > 200:
            color = "Violet"
        elif r > 150 and g < 50 and b > 50:
            color = "Indigo"
        elif r < 50 and g < 50 and b > 200:
            color = "Blue"
        elif r < 50 and g > 200 and b < 50:
            color = "Green"
        elif r > 200 and g > 200 and b < 50:
            color = "Yellow"
        elif r > 200 and g > 100 and b < 50:
            color = "Orange"
        elif r < 50 and g < 50 and b < 50:
            color = "Black"
        elif r > 200 and g > 200 and b > 200:
            color = "White"

        identified_shapes.append({
            "shape": shape,
            "label": label,
            "color": color
        })

    return identified_shapes


def calculate_coordinates(detected_shapes, frame_width, frame_height):
    center_x = frame_width // 2
    center_y = frame_height // 2

    coordinates = []

    for shape in detected_shapes:
        shape_center_x, shape_center_y = shape["center"]

        relative_x = shape_center_x - center_x
        relative_y = center_y - shape_center_y

        coordinates.append((relative_x, relative_y))

    return coordinates


def draw_shapes(image, identified_shapes, coordinates):
    for i, shape_info in enumerate(identified_shapes):
        shape = shape_info["shape"]
        contour = shape["contour"]

        # Draw the contour of the shape
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        label = shape_info["label"]
        color = shape_info["color"]
        center_x, center_y = shape["center"]

        font_scale = 1.5
        font_thickness = 4

        # Add text labels and markers on the shapes
        cv2.putText(image, f"{color} {label}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 255), font_thickness)
        cv2.circle(image, (center_x, center_y), 3, (255, 0, 0), -1)

        x, y = coordinates[i]
        cv2.putText(image, f"({x}, {y})", (center_x, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image


def main():
    camera = cv2.VideoCapture(0)  # Use index 0 for the laptop camera

    while True:
        ret, frame = camera.read()

        detected_shapes = detect_shapes(frame)
        identified_shapes = identify_shapes(detected_shapes, frame)

        frame_height, frame_width, _ = frame.shape
        coordinates = calculate_coordinates(detected_shapes, frame_width, frame_height)

        annotated_frame = draw_shapes(frame, identified_shapes, coordinates)

        cv2.imshow("Shape Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
