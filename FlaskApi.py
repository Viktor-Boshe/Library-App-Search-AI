import jwt
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from functools import lru_cache
import pandas as pd
import pymysql

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = 'b7aef6d0913742a8e5538d9741c92c3f'

global_model = None
global_vectorizer = None
global_book_data = None

@app.before_request
def load_data():
    global global_model, global_vectorizer, global_book_data

    global_model, global_vectorizer = load_model_and_vectorizer()
    global_book_data = load_book_data()

def jwt_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Missing token'}), 401

        try:
            jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        except jwt.exceptions.InvalidTokenError as e:
            return jsonify({'error': str(e)}), 401

        return func(*args, **kwargs)
    return wrapper

@lru_cache(maxsize=1)
def load_model_and_vectorizer():
    model = pickle.load(open('book_search_model.pkl', 'rb'))
    vectorizer = pickle.load(open('book_search_vectorizer.pkl', 'rb'))
    return model, vectorizer

@lru_cache(maxsize=1)
def load_book_data():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='databse',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    with connection.cursor() as cursor:
        query = ("SELECT book_id, book_name, book_author, book_description FROM library WHERE book_name IS NOT NULL "
                 "AND book_author IS NOT NULL AND book_description IS NOT NULL")
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall())

    connection.close()
    return df

@app.route('/api/search', methods=['POST'])
@jwt_required
def search():
    data = request.json
    book_name = data.get('book_name', '')
    book_author = data.get('book_author', '')
    book_description = data.get('book_description', '')

    try:
        query = ' '.join([book_name, book_author, book_description])
        query_vec = global_vectorizer.transform([query])
        distances, indices = global_model.kneighbors(query_vec, n_neighbors=50)

        results = []
        for index in indices[0]:
            book_id = int(global_book_data.iloc[index]['book_id'])
            results.append(book_id)

        return jsonify(results)

    except pymysql.Error as e:
        print(f"Error querying the database: {e}")
        return jsonify({"error": "Error querying the database"}), 500

if __name__ == '__main__':
    app.run()
