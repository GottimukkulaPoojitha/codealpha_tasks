from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from groq import Groq
from faqs import faqs

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)

import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def get_best_answer(user_question):
    try:
        faq_questions = [preprocess(faq['question']) for faq in faqs]
        processed_user_q = preprocess(user_question)
        all_questions = faq_questions + [processed_user_q]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_questions)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        best_match_index = similarity_scores.argmax()
        best_score = similarity_scores[0][best_match_index]

        if best_score >= 0.3:
            return faqs[best_match_index]['answer']

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Answer this question briefly in 2-3 sentences: {user_question}"
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('question', '')
    answer = get_best_answer(user_question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)