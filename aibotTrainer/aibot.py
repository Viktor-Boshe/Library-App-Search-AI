import pymysql
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

try:
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

    df['book_description'] = df['book_description'].astype(str)
    df['book_name'] = df['book_name'].astype(str)
    df['book_author'] = df['book_author'].astype(str)

    df['text'] = df['book_name'] + ' ' + df['book_author'] + ' ' + df['book_name'] + ' ' + df['book_author'] + ' ' + df['book_description']
    df['text'] = df['text'].str.encode('utf-8', errors='ignore').str.decode('utf-8')

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])

    model = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1)
    model.fit(X)

    pickle.dump(model, open('book_search_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('book_search_vectorizer.pkl', 'wb'))

except pymysql.Error as e:
    print(f"Error connecting to the database: {e}")