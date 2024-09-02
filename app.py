from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

nltk.download('punkt')
nltk.download('stopwords')

# Preprocess function for symptom input
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tool')
def tool_interface():
    return render_template('tool_interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')  # Get the symptoms from the form
        
        prediction = model_predict(symptoms)  # Call the prediction function
        return render_template('tool_interface.html', prediction=prediction)

def model_predict(symptoms):
    # Preprocess the symptom input
    preprocessed_symptom = preprocess_text(symptoms)
    
    # Transform the preprocessed symptom using the loaded TF-IDF vectorizer
    symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])
    
    # Make the prediction using the loaded ML model
    predicted_disease = model.predict(symptom_tfidf)
    
    return predicted_disease[0]

if __name__ == '__main__':
    app.run(debug=True)

