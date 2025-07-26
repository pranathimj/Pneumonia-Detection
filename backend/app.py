from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
from predict import predict_pneumonia

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return """
    <h1>Pneumonia Detection System</h1>
    <p>Backend server is running correctly!</p>
    <p>Use the React frontend at <a href="http://localhost:3000">http://localhost:3000</a></p>
    """

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    result, confidence, _ = predict_pneumonia(filepath)
    
    return jsonify({
        "diagnosis": result,
        "confidence": float(confidence),
        "imageUrl": f"/uploads/{filename}"
    })

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    from waitress import serve
    print("\n" + "="*50)
    print("⚡ Production Server Running: http://localhost:5000")
    print("⚡ Frontend Access: http://localhost:3000")
    print("="*50 + "\n")
    serve(app, host="0.0.0.0", port=5000)