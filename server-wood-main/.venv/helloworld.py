from flask import Flask, render_template
from flask import Flask, request, jsonify
import requests
import numpy as np
from PIL import Image

app = Flask(__name__)
         
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/check-wood', methods=['POST'])
def hello_world():
    file = request.files['image']
    img = Image.open(file)

    img_prev = img.resize((224,224))
    img_prev_array = np.array(img_prev) / 255.0
    img_prev_array = np.expand_dims(img_prev_array, axis=0)
    server_prev_url='http://localhost:8501/v1/models/wood1:predict'
    response_prev = requests.post(server_prev_url, json={"instances": img_prev_array.tolist()})
    prediction_prev=response_prev.json()
    res_prev =prediction_prev['predictions'][0]
    max_value=np.max(res_prev)
    if max_value==res_prev[2]:
        img = img.resize((50, 50))
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)
       
        server_url = 'http://localhost:8500/v1/models/wood:predict'  
        response = requests.post(server_url, json={"instances": img_array.tolist()})
        prediction = response.json()
        print(response)
        labs=["Gỗ gió bầu","Gỗ Bạch đàn","Gỗ Lim","Gỗ Sồi","Gỗ thông","Gỗ Trắc(gỗ cầm lai)","Go Tràm","Go xoan"]
        res =prediction['predictions'][0]
        array = [round(num*100, 2) for num in res]
        for i in range(len(array)):
            for j in range(i+1,len(array)):
                if array[i]<array[j]:
                    tam = array[i]
                    array[i]=array[j]
                    
                    array[j]=tam

                    temp = labs[i]
                    labs[i]=labs[j]
                    labs[j]=temp
        
        dbs = [x for x in array if x >= 3]

        resp=[dbs,labs[0:len(dbs)]]
        return resp
    return "Not wood"

@app.route('/phone')
def get_phone():
    return render_template('phone.html')

@app.route('/check-wood')
def showSignUp():
    return render_template('postImage.html')
@app.route('/webcam')
def webcam_page_route():
    return render_template('webcam_page.html')
   
if __name__ == '__main__':
    app.run(port=5000)
    
# http://localhost:5000/check-wood