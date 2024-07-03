from flask import Flask, request, jsonify
import pickle
import pandas as pd
import pymysql
from functools import lru_cache

app = Flask(__name__)

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
        query = "SELECT book_id, book_name, book_author, book_description FROM library WHERE book_name IS NOT NULL AND book_author IS NOT NULL AND book_description IS NOT NULL"
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall())

    connection.close()
    return df

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    book_name = data.get('book_name', '')
    book_author = data.get('book_author', '')
    book_description = data.get('book_description', '')

    try:
        df = load_book_data()
        model, vectorizer = load_model_and_vectorizer()

        query = ' '.join([book_name, book_author, book_description])
        query_vec = vectorizer.transform([query])
        distances, indices = model.kneighbors(query_vec, n_neighbors=50)

        results = []
        for index in indices[0]:
            book_id = int(df.iloc[index]['book_id'])
            results.append(book_id)

        return jsonify(results)

    except pymysql.Error as e:
        print(f"Error querying the database: {e}")
        return jsonify({"error": "Error querying the database"}), 500

if __name__ == '__main__':
    app.run()
