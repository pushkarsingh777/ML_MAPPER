from flask import Flask, request, jsonify, render_template
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Load your model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    data = request.json
    symptom = data['symptom']
    
    # Preprocess the symptom
    preprocessed_symptom = preprocess_text(symptom)
    
    # Transform the preprocessed symptom into TF-IDF features
    symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])
    
    # Predict the disease
    predicted_disease = model.predict(symptom_tfidf)
    
    # Return the predicted disease as a JSON response
    return jsonify({'disease': predicted_disease[0]})

if __name__ == '__main__':
    app.run(debug=True)
