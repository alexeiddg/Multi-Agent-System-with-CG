from flask import Flask, request, Response
import cv2
import numpy as np
from PIL import Image
import io
import webbrowser
import threading
import socket

app = Flask(__name__)
last_frame = None
browser_opened = False  

@app.route('/stream', methods=['POST'])
def stream():
    # Handles receiving a frame via POST, processes it, and saves it for streaming.
    global last_frame, browser_opened
    try:
        # Retrieve image bytes from the request
        image_bytes = request.data
        if not image_bytes:
            return "No image received", 400

        # Convert bytes to a PIL image and then to a NumPy array
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        last_frame = frame

        # Open the browser only once
        if not browser_opened:
            threading.Thread(target=open_browser).start()
            browser_opened = True

        return "Frame received and processed", 200
    except Exception as e:
        return f"Error processing the image: {e}", 500

def generate():
    # Generates frames for streaming as a continuous video feed.
    global last_frame
    while True:
        if last_frame is not None:
            # Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', last_frame)
            frame_bytes = buffer.tobytes()
            # Yield the frame in the required format for the video stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, send a placeholder response
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\nNo frame available\r\n')

@app.route('/video_feed')
def video_feed():
    # Provides the video feed endpoint for clients to access the live stream.
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_local_ip():
    # Gets the local IP address of the machine running the server.
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def open_browser():
    # Opens the default web browser with the correct IP and port to display the video feed.
    ip = get_local_ip()
    url = f"http://{ip}:8000/video_feed"
    print(f"Opening browser at: {url}")
    webbrowser.open(url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
