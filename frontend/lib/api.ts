// Client-side API library — calls Next.js Route Handlers at /api/*
// AWS keys, BEDROCK_MODEL_ID, and OpenSearch credentials are NEVER exposed.
// The Route Handlers proxy to the FastAPI backend server-side.

const API_BASE = ""; // Empty string = same-origin /api/* calls

class ApiService {
    private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
        const url = `${API_BASE}${endpoint}`;
        const config: RequestInit = {
            headers: {
                "Content-Type": "application/json",
                ...options.headers,
            },
            ...options,
        };

        const response = await fetch(url, config);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    /** Semantic search via OpenSearch + Bedrock (POST /api/search) */
    async search(query: string, filters: Record<string, unknown> = {}) {
        return this.request<{ results: Resource[] }>("/api/search", {
            method: "POST",
            body: JSON.stringify({
                query,
                pricing_filter: filters.pricing_filter,
                resource_types: filters.resource_types,
                limit: filters.limit || 20,
            }),
        });
    }

    /** Get all resources with filters (GET /api/resources) */
    async getResources(filters: ResourceFilters = {}) {
        const params = new URLSearchParams();
        if (filters.resource_type) params.append("resource_type", filters.resource_type);
        if (filters.pricing_type) params.append("pricing_type", filters.pricing_type);
        if (filters.limit) params.append("limit", String(filters.limit));
        if (filters.offset) params.append("offset", String(filters.offset));
        return this.request<Resource[]>(`/api/resources?${params}`);
    }

    /** Get single resource (GET /api/resources/[id]) */
    async getResource(id: string) {
        return this.request<Resource>(`/api/resources/${id}`);
    }

    /** Get trending resources sorted by Unified Ranking Algorithm (GET /api/trending) */
    async getTrending(filters: TrendingFilters = {}) {
        const params = new URLSearchParams();
        if (filters.resource_type && filters.resource_type !== "All")
            params.append("resource_type", filters.resource_type);
        if (filters.limit) params.append("limit", String(filters.limit));
        if (filters.pricing_type) params.append("pricing_type", filters.pricing_type);
        if (filters.sort) params.append("sort", filters.sort);
        return this.request<{ results: Resource[] }>(`/api/trending?${params}`);
    }

    /** Health check (GET /api/health) */
    async healthCheck() {
        return this.request<{ status: string }>("/api/health");
    }
}

export interface Resource {
    id: string;
    name: string;
    resource_type: "API" | "Model" | "Dataset";
    description?: string;
    pricing_type?: "free" | "paid" | "freemium";
    github_stars?: number;
    downloads?: number;
    latency_ms?: number;
    p99_latency?: number;
    is_available?: boolean;
    rank?: number;
    stars?: number;
    status?: string;
    install_command?: string;
    endpoint_url?: string;
    docs_url?: string;
    provider?: string;
}

export interface ResourceFilters {
    resource_type?: string | null;
    pricing_type?: string;
    limit?: number;
    offset?: number;
}

export interface TrendingFilters {
    resource_type?: string | null;
    limit?: number;
    pricing_type?: string;
    sort?: string;
}

const apiService = new ApiService();
export default apiService;
