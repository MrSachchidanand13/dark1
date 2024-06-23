import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import base64

app = Flask(__name__)

def process_image(img_stream):
    try:
        # Convert the image stream to a numpy array
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Create a copy of the frame for comparison (simulating video frames)
        frame1 = img.copy()
        
        # Calculate absolute difference between frames
        diff = cv2.absdiff(img, frame1)
        
        # Convert difference to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Convert frame1 to grayscale
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between grayscale frames
        asa = cv2.absdiff(gray, gray_frame1)
        
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(asa, (5, 5), 0)
        
        # Threshold to create a binary image
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY_INV)
        
        # Threshold for another binary image
        _, resh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        
        # Dilate the thresholded images to fill gaps
        dilated = cv2.dilate(thresh, None, iterations=1)
        ilat = cv2.dilate(thresh, None, iterations=1)
        
        # Calculate absolute difference between thresholds
        was = cv2.absdiff(thresh, resh)
        
        # Dilate the difference to emphasize the differences
        lat = cv2.dilate(was, None, iterations=3)
        
        # Find contours in the dilated image with no approximation
        contours, _ = cv2.findContours(ilat, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Find contours in the dilated image with simple approximation
        contour, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original frame
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        
        # Process each contour in the first set of contours
        for c in contours:
            if cv2.contourArea(c) > 100001:
                # Perform some action if the contour area is large
                pass
            elif cv2.contourArea(c) > 100:
                # Perform some action if the contour area is medium
                pass
        
        # Draw contours on the original frame
        cv2.drawContours(img, contours, -1, (255, 0, 255), 1)
        
        
        
        # Encode processed image to base64 string
        _, encoded_img = cv2.imencode('.jpg', img)
        base64_img = base64.b64encode(encoded_img).decode('utf-8')

        return base64_img

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        
        # Process the uploaded image to grayscale
        processed_image = process_image(file)
        
        if processed_image:
            return jsonify({
                'success': True,
                'processed_image': processed_image
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to process image'})
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'message': 'Error processing image'})

if __name__ == '__main__':
    app.run(debug=True)
