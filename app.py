from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the text classification model
classifier = pipeline("text-classification", model="ZidanAf/Zidan_model_output_v4")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = classifier(text)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
