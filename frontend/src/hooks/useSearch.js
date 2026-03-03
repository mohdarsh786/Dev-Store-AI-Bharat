import { useState, useCallback } from 'react';
import apiService from '../services/api';

export function useSearch() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const search = useCallback(async (query, filters = {}) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.search(query, filters);
      setResults(response.results || []);
    } catch (err) {
      setError(err.message);
      // Fallback to mock data if API fails
      setResults([
        {
          id: '1',
          name: 'OpenAI GPT-4 API',
          description: 'Advanced language model API for natural language processing',
          resource_type: 'API',
          pricing_type: 'paid',
          score: 0.95,
          github_stars: 50000,
          downloads: 1000000
        },
        {
          id: '2',
          name: 'Hugging Face Transformers',
          description: 'State-of-the-art machine learning models',
          resource_type: 'Model',
          pricing_type: 'free',
          score: 0.92,
          github_stars: 75000,
          downloads: 2500000
        },
        {
          id: '3',
          name: 'Common Crawl Dataset',
          description: 'Petabyte-scale web crawl data',
          resource_type: 'Dataset',
          pricing_type: 'free',
          score: 0.88,
          downloads: 500000
        }
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  return {
    results,
    loading,
    error,
    search,
    clearResults
  };
}