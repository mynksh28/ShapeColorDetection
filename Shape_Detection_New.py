import cv2
import numpy as np
import json

color_ranges = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "green": ([35, 50, 50], [85, 255, 255]),
    "blue": ([100, 100, 100], [130, 255, 255]),
    "violet": ([130, 100, 100], [160, 255, 255]),
    "indigo": ([100, 0, 0], [120, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255]),
    "orange": ([5, 100, 100], [15, 255, 255]),
    "black": ([0, 0, 0], [180, 255, 30]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "pink": ([140, 100, 100], [170, 255, 255]),
    "brown": ([0, 60, 60], [20, 255, 255]),
    "grey": ([0, 0, 40], [180, 30, 190])
}

color_mapping = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "violet": (255, 0, 255),
    "indigo": (75, 0, 130),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "grey": (128, 128, 128)
}

# Set the minimum area for shape detection
MIN_SHAPE_AREA = 1200

# Function to detect shapes in an image
def detect_shapes(image):
    # Convert image to grayscale
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
                "contour": contour.tolist(),
                "num_vertices": num_vertices,
                "center": (center_x, center_y),
                "area": area,
                "circularity": circularity,
                "coordinates": (center_x, center_y)
            }

            detected_shapes.append(shape)

    return detected_shapes

# Function to identify color based on pixel value
def identify_color_hsv(pixel_color):
    hsv_color = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)[0][0]
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        if all(lower <= c <= upper for c, lower, upper in zip(hsv_color, lower_bound, upper_bound)):
            return color_name
    return "Unknown"

def identify_shapes_and_colors(detected_shapes, image):
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
            x, y, width, height = cv2.boundingRect(np.array(shape["contour"]))
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
        red, green, blue = pixel_color[:3]

        # Color detection
        for color_name, (lower_bound, upper_bound) in color_ranges.items():
            if all(lower <= c <= upper for c, lower, upper in zip((blue, green, red), lower_bound, upper_bound)):
                color = color_name
                break
        else:
            color = "Unknown"

        identified_shapes.append({
            "shape": shape,
            "label": label,
            "color": color
        })

    return identified_shapes

def main():
    # Open the camera (use index 0 for the default camera)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Failed to open the camera.")
        return

    data = []

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to retrieve frame from the camera. Exiting.")
            break

        detected_shapes = detect_shapes(frame)
        identified_shapes = identify_shapes_and_colors(detected_shapes, frame)

        # Calculate the center of the camera's view
        frame_height, frame_width, _ = frame.shape
        center_x = frame_width // 2
        center_y = frame_height // 2

        for shape_info in identified_shapes:
            shape = shape_info["shape"]
            label = shape_info["label"]

            # Check if "center" key exists in shape_info
            if "center" in shape:
                shape_center_x, shape_center_y = shape["center"]

                # Calculate relative coordinates based on the center of the camera's view
                relative_x = shape_center_x - center_x
                relative_y = center_y - shape_center_y

                color_name = identify_color_hsv(frame[shape_center_y, shape_center_x])

                # If color is 'Unknown', use default color (black)
                color = color_mapping.get(color_name, (0, 0, 0))

                # Draw the contour of the shape with the identified color
                cv2.drawContours(frame, [np.array(shape["contour"])], -1, color, 2)

                # Add text labels and markers on the shapes
                cv2.putText(frame, f"{color_name.capitalize()} {label}", (shape_center_x, shape_center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                cv2.circle(frame, (shape_center_x, shape_center_y), 3, (255, 0, 0), -1)

                # Display the relative coordinates
                cv2.putText(frame, f"({relative_x}, {relative_y})", (shape_center_x, shape_center_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 , 0), 2)

        # Display the annotated frame and handle user input
        cv2.imshow("Shape Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    # Save the identified shapes data to a JSON file
    for shape_info in identified_shapes:
        shape = shape_info["shape"]
        label = shape_info["label"]
        color_name = shape_info["color"]
        center = shape.get("center", (0, 0))  # Use default (0, 0) if "center" key is missing

        data.append({
            "shape": shape,
            "label": label,
            "color": color_name,
            "coordinates": center
        })

    with open("shape_data.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    main()
