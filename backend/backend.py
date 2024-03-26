from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Sample data for demonstration
sample_data = [
    {'Title': 'Sample Dress 1', 'Description': 'This is a sample dress with floral print.'},
    {'Title': 'Sample Dress 2', 'Description': 'Another sample dress with stripes.'},
    {'Title': 'Sample Skirt', 'Description': 'A sample skirt with pleats.'}
]

def scrape_all_pages(base_url, max_pages):
    items_data = []

    page_number = 1
    while not max_pages or page_number <= max_pages:
        url = f'{base_url}&page={page_number}'
        result = requests.get(url)
        content = result.text
        soup = BeautifulSoup(content, 'html.parser')

        item_links = [a['href'] for a in soup.find_all('a', class_='tile__covershot')]

        for item_link in item_links:
            item_url = f'https://poshmark.com{item_link}'
            item_result = requests.get(item_url)
            item_content = item_result.text
            item_soup = BeautifulSoup(item_content, 'html.parser')

            title_elem = item_soup.find('h1', class_='fw--light m--r--2 listing__title-container')
            desc_elem = item_soup.find('div', class_='listing__description fw--light')

            title = title_elem.get_text(strip=True, separator=' ') if title_elem else 'N/A'
            desc = desc_elem.get_text(strip=True, separator=' ') if desc_elem else 'N/A'

            items_data.append({'Title': title, 'Description': desc, 'Url': item_url})

        page_number += 1

    return items_data

def get_similar_items(query, data):
    # Combine title and description for TF-IDF vectorization
    text_data = [item['Title'] + ' ' + item['Description'] for item in data]
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query vector and item vectors
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    
    # Get indices of similar items based on similarity scores
    similar_indices = cosine_similarities.argsort()[:-4:-1]  # Get top 3 similar items
    
    similar_items = [{'Title': data[i]['Title'], 'Description': data[i]['Description'], 'Url': data[i]['Url']} for i in similar_indices]
    return similar_items

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/search', methods=['POST'])
def search():
    base_url = 'https://poshmark.com/category/Women-Dresses?sort_by=relevance_v2&condition=nwt_and_ret'
    max_pages_to_scrape = 1  # Set the maximum number of pages to scrape, or None for all pages
    items_data = scrape_all_pages(base_url, max_pages=max_pages_to_scrape)

    query = request.form.get('query', '')  # Get query from form, default to empty string
    if not query:
        return jsonify({'error': 'Query parameter is missing.'}), 400

    similar_items = get_similar_items(query, items_data)
    return jsonify(similar_items)

if __name__ == "__main__":
    app.run(debug=True)
