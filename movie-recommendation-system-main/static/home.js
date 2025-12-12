// TMDb API configuration
const api_key = '47b60aaf43a6f85780c217395976aee5';
const base_url = 'https://api.themoviedb.org/3';
const img_url = 'https://image.tmdb.org/t/p/w500';

let currentGenre = '';

// Fetch and display trending movies
async function loadTrendingMovies(genre = '') {
  try {
    let url = `${base_url}/trending/movie/week?api_key=${api_key}`;
    
    const response = await fetch(url);
    const data = await response.json();
    
    let movies = data.results;
    
    // Filter by genre if selected
    if (genre) {
      movies = movies.filter(movie => movie.genre_ids.includes(parseInt(genre)));
    }
    
    displayMovies(movies);
  } catch (error) {
    console.error('Error loading trending movies:', error);
  }
}

// Display movies in grid
function displayMovies(movies) {
  const container = document.getElementById('trendingMovies');
  container.innerHTML = '';
  
  movies.forEach(movie => {
    const movieCard = document.createElement('div');
    movieCard.style.cssText = 'cursor: pointer; transition: transform 0.3s ease; border-radius: 12px; overflow: hidden; background-color: hsl(var(--card));';
    movieCard.onmouseover = function() { this.style.transform = 'scale(1.05)'; };
    movieCard.onmouseout = function() { this.style.transform = 'scale(1)'; };
    
    movieCard.innerHTML = `
      <img src="${img_url}${movie.poster_path}" alt="${movie.title}" style="width: 100%; height: 300px; object-fit: cover;">
      <div style="padding: 15px;">
        <h3 style="font-size: 1rem; margin: 0; color: hsl(var(--foreground)); font-weight: 600;">${movie.title}</h3>
        <p style="font-size: 0.875rem; opacity: 0.7; margin-top: 5px; color: hsl(var(--foreground));">⭐ ${movie.vote_average.toFixed(1)}</p>
      </div>
    `;
    
    movieCard.onclick = function() {
      document.getElementById('autoComplete').value = movie.title;
      document.querySelector('.movie-button').disabled = false;
      document.querySelector('.movie-button').click();
    };
    
    container.appendChild(movieCard);
  });
}

// Load genres
async function loadGenres() {
  try {
    const response = await fetch(`${base_url}/genre/movie/list?api_key=${api_key}`);
    const data = await response.json();
    
    const genreButtons = document.getElementById('genreButtons');
    
    data.genres.slice(0, 10).forEach(genre => {
      const btn = document.createElement('button');
      btn.className = 'genre-btn';
      btn.dataset.genre = genre.id;
      btn.textContent = genre.name;
      btn.style.cssText = 'background-color: rgba(255,255,255,0.1); border: 1px solid hsl(var(--border)); color: hsl(var(--foreground)); padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.3s ease;';
      
      btn.onclick = function() {
        currentGenre = this.dataset.genre;
        document.querySelectorAll('.genre-btn').forEach(b => {
          b.style.backgroundColor = 'rgba(255,255,255,0.1)';
          b.style.borderColor = 'hsl(var(--border))';
        });
        this.style.backgroundColor = 'hsl(var(--primary))';
        this.style.borderColor = 'hsl(var(--primary))';
        loadTrendingMovies(currentGenre);
      };
      
      genreButtons.appendChild(btn);
    });
  } catch (error) {
    console.error('Error loading genres:', error);
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  loadTrendingMovies();
  loadGenres();
  
  // Check if we need to load a movie from watchlist
  const loadMovieTitle = sessionStorage.getItem('loadMovieTitle');
  if (loadMovieTitle) {
    sessionStorage.removeItem('loadMovieTitle');
    sessionStorage.removeItem('loadMovieId');
    // Set the movie title in search and trigger search
    setTimeout(() => {
      document.getElementById('autoComplete').value = loadMovieTitle;
      document.querySelector('.movie-button').disabled = false;
      document.querySelector('.movie-button').click();
    }, 500);
  }
  
  // Reset genre filter when clicking "All"
  document.querySelector('.genre-btn[data-genre=""]').onclick = function() {
    currentGenre = '';
    document.querySelectorAll('.genre-btn').forEach(b => {
      b.style.backgroundColor = 'rgba(255,255,255,0.1)';
      b.style.borderColor = 'hsl(var(--border))';
    });
    this.style.backgroundColor = 'hsl(var(--primary))';
    this.style.borderColor = 'hsl(var(--primary))';
    loadTrendingMovies();
  };
});
