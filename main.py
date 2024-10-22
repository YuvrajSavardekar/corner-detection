import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Helper function to process video and apply Harris Corner Detection
def process_video(file_path):
    # Open the video file
    print("Atleast made it here, 1")
    cap = cv2.VideoCapture(file_path)
    corner_data = []
    print("Atleast made it here, 2")
    
    if not cap.isOpened():
        return {"error": "Cannot open video file"}
    print("Atleast made it here, 3")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        # Harris corner detection
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        
        # Threshold for an optimal value, marking corners
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark detected corners in red
        
        # Extract the corner coordinates
        corners = np.argwhere(dst > 0.01 * dst.max())
        # Vectorized approach to convert to list of dictionaries
        corner_data = np.column_stack((corners[:, 1], corners[:, 0])).tolist()

        # Format the list as a JSON-serializable structure
        corner_data_json = [{"x": int(x), "y": int(y)} for x, y in corner_data]

    cap.release()
    return {"corners": corner_data_json}

# API route to upload video and return corner detection result
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary location
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)
    
    # Process the video file
    result = process_video(temp_path)
    
    # Clean up the saved video
    os.remove(temp_path)

    return jsonify(result)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
