from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/check-wood', methods=['POST'])
def check_wood():
    file = request.files['image']
    img = Image.open(file).convert("RGB")

    # Preprocess for wood1 model (ไม้/ไม่ใช่ไม้)
    img_prev = img.resize((224, 224))
    img_prev_array = np.array(img_prev).astype(np.float32) / 255.0
    img_prev_array = np.expand_dims(img_prev_array, axis=0)

    # Model 1 - ตรวจว่าเป็นไม้หรือไม่
    server_prev_url = 'http://localhost:8501/v1/models/wood1:predict'
    response_prev = requests.post(server_prev_url, json={"instances": img_prev_array.tolist()})
    prediction_prev = response_prev.json()

    try:
        res_prev = prediction_prev['predictions'][0]
        print("Prediction from wood1:", res_prev)

        # เช็คว่าโมเดลมั่นใจว่าเป็น "ไม้" (class 2)
        if np.argmax(res_prev) == 2 and res_prev[2] >= 0.6:
            # Preprocess for model 2 (จำแนกชนิดไม้)
            img_wood = img.resize((50, 50))
            img_array = np.array(img_wood).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            server_url = 'http://localhost:8500/v1/models/wood:predict'
            response = requests.post(server_url, json={"instances": img_array.tolist()})
            prediction = response.json()
            print("Prediction from wood:", prediction)

            res = prediction['predictions'][0]
            labs = ["Gỗ gió bầu", "Gỗ Bạch đàn", "Gỗ Lim", "Gỗ Sồi", "Gỗ thông", "Gỗ Trắc(gỗ cầm lai)", "Gỗ Tràm", "Gỗ xoan"]
            array = [round(num * 100, 2) for num in res]

            # จัดเรียงค่าจากมากไปน้อย
            sorted_results = sorted(zip(array, labs), reverse=True)
            array_sorted, labs_sorted = zip(*sorted_results)

            # กรองเฉพาะที่มีโอกาสมากกว่า 3%
            dbs = [x for x in array_sorted if x >= 3]
            top_labs = labs_sorted[:len(dbs)]

            return jsonify([dbs, top_labs])
        else:
            return jsonify("Not wood")
    except KeyError:
        return jsonify({"error": "Model wood1 response malformed", "details": prediction_prev})

@app.route('/phone')
def get_phone():
    return render_template('phone.html')

@app.route('/webcam')
def webcam_page_route():
    return render_template('webcam_page.html')

@app.route('/check-wood')
def showSignUp():
    return render_template('postImage.html')

if __name__ == '__main__':
    app.run(port=5000)
