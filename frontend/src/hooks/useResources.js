import { useState, useEffect } from 'react';
import apiService from '../services/api';

export function useResources(type = null) {
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResources = async () => {
      try {
        setLoading(true);
        const data = await apiService.getResources({ 
          resource_type: type,
          limit: 10 
        });
        setResources(data);
      } catch (err) {
        setError(err.message);
        // Fallback mock data
        const mockData = {
          'API': [
            { id: '1', name: 'Azure OpenAI', version: 'v4.0', rating: 4.9, pricing_type: 'paid', initial: 'A' },
            { id: '2', name: 'Stable Video', version: 'v1.1', rating: 4.7, pricing_type: 'freemium', initial: 'S' }
          ],
          'Model': [
            { id: '3', name: 'Llama-3-70b', version: 'Quantized', downloads: '2.1M', pricing_type: 'free', initial: 'L' },
            { id: '4', name: 'Mistral Large', version: 'Instruct', downloads: '800k', pricing_type: 'free', initial: 'M' }
          ],
          'Dataset': [
            { id: '5', name: 'Common Crawl (Hi)', size: '500GB', tag: 'Cleaned', pricing_type: 'free', icon: 'storage' },
            { id: '6', name: 'LAION Indian Art', size: '12GB', tag: 'Tagged', pricing_type: 'free', icon: 'image' }
          ]
        };
        setResources(mockData[type] || []);
      } finally {
        setLoading(false);
      }
    };

    fetchResources();
  }, [type]);

  return { resources, loading, error };
}