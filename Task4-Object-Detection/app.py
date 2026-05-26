from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('yolov8n.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    detected_objects = []

    if request.method == 'POST':
        file = request.files['video']
        if file:
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            cap = cv2.VideoCapture(filepath)
            ret, frame = cap.read()
            cap.release()

            if ret:
                results = model(frame)
                annotated_frame = results[0].plot()

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    confidence = float(box.conf[0])
                    detected_objects.append({
                        'label': label,
                        'confidence': round(confidence * 100, 2)
                    })

                result_filename = 'result_' + filename.split('.')[0] + '.jpg'
                result_path = os.path.join(UPLOAD_FOLDER, result_filename)
                cv2.imwrite(result_path, annotated_frame)
                result_image = result_filename

    return render_template('index.html',
                         result_image=result_image,
                         detected_objects=detected_objects)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)