import { useState } from 'react';
import SearchBar from '../components/SearchBar';
import ResourceCard from '../components/ResourceCard';
import ThemeToggle from '../components/ThemeToggle';

export default function HomePage() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (query) => {
    setLoading(true);
    try {
      // Mock data for now
      setTimeout(() => {
        setResults([
          {
            id: '1',
            name: 'OpenAI GPT-4 API',
            description: 'Advanced language model API for natural language processing, text generation, and conversational AI applications.',
            resource_type: 'API',
            pricing_type: 'paid',
            score: 0.95
          },
          {
            id: '2',
            name: 'Hugging Face Transformers',
            description: 'State-of-the-art machine learning models for NLP tasks including text classification, translation, and question answering.',
            resource_type: 'Model',
            pricing_type: 'free',
            score: 0.92
          },
          {
            id: '3',
            name: 'Common Crawl Dataset',
            description: 'Petabyte-scale web crawl data containing billions of web pages for training large language models and research.',
            resource_type: 'Dataset',
            pricing_type: 'free',
            score: 0.88
          },
          {
            id: '4',
            name: 'Anthropic Claude API',
            description: 'Constitutional AI assistant with advanced reasoning capabilities and safety features for enterprise applications.',
            resource_type: 'API',
            pricing_type: 'paid',
            score: 0.87
          },
          {
            id: '5',
            name: 'Stable Diffusion Models',
            description: 'Open-source text-to-image generation models for creating high-quality images from text descriptions.',
            resource_type: 'Model',
            pricing_type: 'free',
            score: 0.85
          }
        ]);
        setLoading(false);
      }, 800);
    } catch (error) {
      console.error('Search failed:', error);
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="app-header">
          <h1>DevStore</h1>
          <ThemeToggle />
        </header>

        <div className="glass-card search-section">
          <div className="hero-section">
            <h2>Discover Developer Resources</h2>
            <p>
              Search for APIs, Models, and Datasets powered by AI. 
              Find the perfect tools for your next project.
            </p>
          </div>
          <SearchBar onSearch={handleSearch} />
        </div>

        {loading && (
          <div className="glass-card loading-state">
            <div className="loading-spinner"></div>
            <p>Searching for resources...</p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div>
            <div className="results-header">
              <h3>Search Results</h3>
              <span className="results-count">{results.length} resources found</span>
            </div>
            {results.map(resource => (
              <ResourceCard key={resource.id} resource={resource} />
            ))}
          </div>
        )}

        {!loading && results.length === 0 && (
          <div className="glass-card empty-state">
            <div className="empty-state-icon">🔍</div>
            <h3>Start Your Search</h3>
            <p>
              Enter a query above to discover amazing developer resources. 
              Try searching for "language models", "image APIs", or "datasets".
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
