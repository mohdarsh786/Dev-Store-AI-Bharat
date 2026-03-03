// API service for DevStore
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Search resources
  async search(query, filters = {}) {
    return this.request('/api/v1/search', {
      method: 'POST',
      body: JSON.stringify({
        query,
        pricing_filter: filters.pricing_filter,
        resource_types: filters.resource_types,
        limit: filters.limit || 20
      })
    });
  }

  // Get all resources
  async getResources(filters = {}) {
    const params = new URLSearchParams();
    if (filters.resource_type) params.append('resource_type', filters.resource_type);
    if (filters.pricing_type) params.append('pricing_type', filters.pricing_type);
    if (filters.limit) params.append('limit', filters.limit);
    if (filters.offset) params.append('offset', filters.offset);

    return this.request(`/api/v1/resources?${params}`);
  }

  // Get single resource
  async getResource(id) {
    return this.request(`/api/v1/resources/${id}`);
  }

  // Health check
  async healthCheck() {
    return this.request('/api/v1/health');
  }
}

export default new ApiService();