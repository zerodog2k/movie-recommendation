from dotenv import load_dotenv
load_dotenv()
from email.mime import application
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bs4 import BeautifulSoup
import pickle
import requests
import os
from pymongo import MongoClient
from bson.objectid import ObjectId

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# MongoDB Atlas connection
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI is missing from your .env file")

try:
    client = MongoClient(MONGODB_URI)
    db = client['movie_recommendation']
    users_collection = db['users']
    watchlists_collection = db['watchlists']
    reviews_collection = db['reviews']
    print("Connected to MongoDB successfully")
except Exception as e:
    print("❌ MongoDB Connection Failed:", e)
    raise SystemExit("Fix your MongoDB connection and restart.")

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id, email, name):
        self.id = str(user_id)
        self.email = email
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data['_id'], user_data['email'], user_data['name'])
    return None

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not name or not email or not password:
            flash('All fields are required', 'signup_error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'signup_error')
            return render_template('signup.html')
        
        # Password complexity validation
        import re
        if not re.search(r'[A-Z]', password):
            flash('Password must contain at least one uppercase letter', 'signup_error')
            return render_template('signup.html')
        if not re.search(r'[0-9]', password):
            flash('Password must contain at least one number', 'signup_error')
            return render_template('signup.html')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            flash('Password must contain at least one special character (!@#$%^&*(),.?":{}|<>)', 'signup_error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'signup_error')
            return render_template('signup.html')
        
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            flash('Email already registered. Please login.', 'signup_error')
            return render_template('signup.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        user_data = {
            "name": name,
            "email": email,
            "password": hashed_password
        }
        
        result = users_collection.insert_one(user_data)
        
        flash('Account created successfully! Please login.', 'signup_success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'login_error')
            return render_template('login.html')
        
        user_data = users_collection.find_one({"email": email})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data['_id'], user_data['email'], user_data['name'])
            login_user(user)
            flash('Logged in successfully!', 'login_success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'login_error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route("/watchlist")
@login_required
def watchlist():
    user_watchlists = watchlists_collection.find({"user_id": current_user.id})
    
    to_watch = []
    watched = []
    
    for item in user_watchlists:
        movie_data = {
            'movie_id': item['movie_id'],
            'title': item['title'],
            'poster_path': item['poster_path'],
            'rating': item.get('rating', 'N/A')
        }
        
        if item['list_type'] == 'to_watch':
            to_watch.append(movie_data)
        elif item['list_type'] == 'watched':
            watched.append(movie_data)
    
    return render_template('watchlist.html', to_watch=to_watch, watched=watched)

@app.route("/add_to_watchlist", methods=["POST"])
@login_required
def add_to_watchlist():
    movie_id = request.form.get('movie_id')
    title = request.form.get('title')
    poster_path = request.form.get('poster_path')
    rating = request.form.get('rating')
    list_type = request.form.get('list_type')  # 'to_watch' or 'watched'
    
    # Check if already in watchlist
    existing = watchlists_collection.find_one({
        "user_id": current_user.id,
        "movie_id": movie_id,
        "list_type": list_type
    })
    
    if not existing:
        watchlists_collection.insert_one({
            "user_id": current_user.id,
            "movie_id": movie_id,
            "title": title,
            "poster_path": poster_path,
            "rating": rating,
            "list_type": list_type
        })
        return json.dumps({'success': True, 'message': f'Added to {list_type.replace("_", " ")} list!'})
    else:
        return json.dumps({'success': False, 'message': 'Movie already in your list'}), 400

@app.route("/remove_from_watchlist", methods=["POST"])
@login_required
def remove_from_watchlist():
    movie_id = request.form.get('movie_id')
    list_type = request.form.get('list_type')
    
    watchlists_collection.delete_one({
        "user_id": current_user.id,
        "movie_id": movie_id,
        "list_type": list_type
    })
    
    flash('Removed from watchlist', 'success')
    return redirect(url_for('watchlist'))

@app.route("/add_review", methods=["POST"])
@login_required
def add_review():
    imdb_id = request.form.get('imdb_id')
    review_text = request.form.get('review_text')
    movie_title = request.form.get('movie_title')
    
    if not review_text or len(review_text.strip()) < 10:
        return json.dumps({'success': False, 'message': 'Review must be at least 10 characters long'}), 400
    
    # Check if user already reviewed this movie
    existing_review = reviews_collection.find_one({
        "user_id": current_user.id,
        "imdb_id": imdb_id
    })
    
    if existing_review:
        return json.dumps({'success': False, 'message': 'You have already reviewed this movie'}), 400
    
    # Add review to database
    reviews_collection.insert_one({
        "user_id": current_user.id,
        "user_name": current_user.name,
        "imdb_id": imdb_id,
        "movie_title": movie_title,
        "review_text": review_text.strip()
    })
    
    # Apply sentiment analysis
    movie_review_array = np.array([review_text.strip()])
    movie_vector = vectorizer.transform(movie_review_array)
    pred = clf.predict(movie_vector)
    sentiment = 'Good' if pred else 'Bad'
    
    return json.dumps({
        'success': True, 
        'message': 'Review added successfully!',
        'review': {
            'text': review_text.strip(),
            'sentiment': sentiment,
            'user_name': current_user.name,
            'review_id': str(reviews_collection.find_one({"user_id": current_user.id, "imdb_id": imdb_id})['_id'])
        }
    })

@app.route("/edit_review", methods=["POST"])
@login_required
def edit_review():
    review_id = request.form.get('review_id')
    review_text = request.form.get('review_text')
    
    if not review_text or len(review_text.strip()) < 10:
        return json.dumps({'success': False, 'message': 'Review must be at least 10 characters long'}), 400
    
    # Check if review belongs to current user
    review = reviews_collection.find_one({"_id": ObjectId(review_id), "user_id": current_user.id})
    if not review:
        return json.dumps({'success': False, 'message': 'Review not found or unauthorized'}), 404
    
    reviews_collection.update_one(
        {"_id": ObjectId(review_id)},
        {"$set": {"review_text": review_text.strip()}}
    )
    
    # Apply sentiment analysis
    movie_review_array = np.array([review_text.strip()])
    movie_vector = vectorizer.transform(movie_review_array)
    pred = clf.predict(movie_vector)
    sentiment = 'Good' if pred else 'Bad'
    
    return json.dumps({
        'success': True,
        'message': 'Review updated successfully!',
        'sentiment': sentiment
    })

@app.route("/delete_review", methods=["POST"])
@login_required
def delete_review():
    review_id = request.form.get('review_id')
    
    # Check if review belongs to current user
    review = reviews_collection.find_one({"_id": ObjectId(review_id), "user_id": current_user.id})
    if not review:
        return json.dumps({'success': False, 'message': 'Review not found or unauthorized'}), 404
    
    reviews_collection.delete_one({"_id": ObjectId(review_id)})
    
    return json.dumps({'success': True, 'message': 'Review deleted successfully!'})

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']
    trailer_key = request.form.get('trailer_key', '')
    teaser_key = request.form.get('teaser_key', '')
    movie_id = request.form.get('movie_id', '')
    poster_path = request.form.get('poster_path', '')

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}
    print(f"calling imdb api: {'https://www.imdb.com/title/{}/reviews/?ref_=tt_ov_rt'.format(imdb_id)}")
    
    # Get user reviews from MongoDB with review IDs
    user_reviews_list = list(reviews_collection.find({"imdb_id": imdb_id}))
    user_reviews = {}
    
    for review in user_reviews_list:
        review_text = review['review_text']
        # Apply sentiment analysis to user reviews
        movie_review_array = np.array([review_text])
        movie_vector = vectorizer.transform(movie_review_array)
        pred = clf.predict(movie_vector)
        sentiment = 'Good' if pred else 'Bad'
        user_name = review.get('user_name', 'Anonymous')
        review_id = str(review['_id'])
        user_id = review.get('user_id', '')
        user_reviews[review_text] = {
            'sentiment': sentiment,
            'user_name': user_name,
            'review_id': review_id,
            'user_id': user_id
        }
    
    # web scraping to get user reviews from IMDB site
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}

    response = requests.get(url, headers=headers)
    print(response.status_code)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')
        soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})
        print(soup_result)

        reviews_list = [] # list of reviews
        reviews_status = [] # list of comments (good or bad)
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # passing the review to our model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Good (from IMDB)' if pred else 'Bad (from IMDB)')

        # combining reviews and comments into a dictionary
        scraped_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
        
        # Convert scraped reviews to same format
        scraped_reviews_formatted = {}
        for text, status in scraped_reviews.items():
            scraped_reviews_formatted[text] = {
                'sentiment': 'Good' if 'Good' in status else 'Bad',
                'user_name': 'IMDB User',
                'review_id': '',
                'user_id': '',
                'is_scraped': True
            }
        
        # Combine user reviews and scraped reviews
        all_reviews = {**user_reviews, **scraped_reviews_formatted}

        # passing all the data to the html file
        return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
            vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
            movie_cards=movie_cards,reviews=all_reviews,casts=casts,cast_details=cast_details,
            trailer_key=trailer_key,teaser_key=teaser_key,movie_id=movie_id,poster_path=poster_path,imdb_id=imdb_id)
    else:
        print("Failed to retrieve reviews")
        return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
            vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
            movie_cards=movie_cards,reviews=user_reviews,casts=casts,cast_details=cast_details,
            trailer_key=trailer_key,teaser_key=teaser_key,movie_id=movie_id,poster_path=poster_path,imdb_id=imdb_id)

if __name__ == '__main__':
    app.run(debug=True)
