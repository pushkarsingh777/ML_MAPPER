from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


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
        symptoms = request.form.get('symptoms')  
        
        prediction = model_predict(symptoms) 
        return render_template('tool_interface.html', prediction=prediction)

def model_predict(symptoms):
    
    preprocessed_symptom = preprocess_text(symptoms)
    
    
    symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])
    
   
    predicted_disease = model.predict(symptom_tfidf)
    
    return predicted_disease[0]

if __name__ == '__main__':
    app.run(debug=True)

