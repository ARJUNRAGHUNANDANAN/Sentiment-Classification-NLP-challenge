from flask import Flask, render_template, request, jsonify
import os
import json
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    review = request.form['review']
    # get from https://cloud.google.com/python/docs/reference/aiplatform/latest/index.html
    # sampe request
    ENDPOINT_ID="<replace with your endpoint id>"
    PROJECT_ID="<replace with your endpoint project id>"
    data = {
        "instances": [
            { "text": review }  # Refer to index.html template to see how to enter the review.
        ]
    }

    access_token = os.popen('gcloud auth print-access-token').read().strip()
    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}:predict"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        predictions = response.json()['predictions']
        score = predictions[0][0] 
        result = "Positive" if score > 0 else "Negative"
    else:
        print(f"ENDPOINT_ID: {os.environ.get('ENDPOINT_ID')}")
        print(f"PROJECT_ID: {os.environ.get('PROJECT_ID')}")
        print(f"Error: {response.status_code} - {response.text}")
        print(access_token)
        result = 'Error calling Vertex AI endpoint' 
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)