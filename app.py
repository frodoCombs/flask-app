from flask import Flask, render_template, request, jsonify
import joblib
import pickle
from transformers import DistilBertTokenizer, TextClassificationPipeline, TFDistilBertForSequenceClassification
import sys

app = Flask(__name__)

# Load your machine learning model (replace 'model.pkl' with your model file)
# model = joblib.load('model.pkl')

# Load model
model = TFDistilBertForSequenceClassification.from_pretrained('model/clf')
model_name, max_len = pickle.load(open('model/info.pkl', 'rb'))
tkzr = DistilBertTokenizer.from_pretrained(model_name)
pipe = TextClassificationPipeline(model=model, tokenizer=tkzr, return_all_scores=True)
stopwords = []
for line in open("stopwords.txt","r"):
    stopwords.append(line.strip())
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    input_text = request.json['input_text']
    result = classify_input(input_text)
    return jsonify({'result': result})

def classify_input(input_text):
    cleaned_text = [e for e in input_text.replace("\n","").lower().split(" ") if e.lower() not in stopwords]
    print("Classifying", file=sys.stderr)
    predictions = []
    for i in range(0,len(cleaned_text),100):
        # print(f"{i}: {cleaned_text[i]}", file=sys.stderr)
        predictions.append(pipe(' '.join(cleaned_text[i:i+100])))

    return convert_prediction_to_string(predictions)

def convert_prediction_to_string(preds):
    d_score = max([pred[0][0]["score"] for pred in preds])
    r_score = max([pred[0][1]["score"] for pred in preds])
    slant = "Left" if d_score > r_score else "Right"
    slant_score = max(d_score, r_score)
    if slant_score < 0.6:
        pred = "Pretty Neutral"
    elif slant_score < 0.7:
        pred = f"Kinda {slant.lower()} leaning"
    elif slant_score < 0.8:
        pred = f"{slant} leaning"
    else:
        pred = f"Very {slant.lower()} leaning"
    return pred

if __name__ == '__main__':
    app.run(debug=True)
