import { useState } from 'react';
import './SearchBar.css';

export default function SearchBar({ onSearch }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="search-form">
      <div className="search-input-wrapper">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          className="glass-input search-input"
          placeholder="Search for APIs, Models, Datasets..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" className="glass-button search-button">
          Search
        </button>
      </div>
    </form>
  );
}
