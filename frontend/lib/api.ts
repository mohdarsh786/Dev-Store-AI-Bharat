// Client-side API library — calls Next.js Route Handlers or FastAPI directly (if NEXT_PUBLIC_API_BASE_URL is set)
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

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

    /** Search resources by query string via the resources route handler. */
    async search(query: string, filters: Record<string, unknown> = {}) {
        const payload = {
            query,
            pricing_filter: filters.pricing_filter || null,
            resource_types: filters.resource_types || null,
            limit: filters.limit || 40,
        };
        const response = await this.request<{ results: Resource[] }>("/api/v1/search", {
            method: "POST",
            body: JSON.stringify(payload),
        });

        return this.deduplicateResults(response);
    }

    /** AI Intent search using Bedrock embeddings + k-NN (POST /api/search/intent) */
    async intentSearch(query: string, filters: Record<string, unknown> = {}) {
        const response = await this.request<{ results: Resource[] }>("/api/v1/search/intent", {
            method: "POST",
            body: JSON.stringify({
                query,
                pricing_filter: filters.pricing_filter,
                resource_types: filters.resource_types,
                limit: filters.limit || 40,
            }),
        });

        return this.deduplicateResults(response);
    }

    private deduplicateResults(response: any) {
        if (!response.results) return response;
        
        const seenNames = new Set<string>();
        const dedupedResults: Resource[] = [];
        
        const CATEGORY_MAP: Record<string, string> = { 'model': 'Model', 'api': 'API', 'dataset': 'Dataset' };
        
        for (const res of response.results) {
            // Production-grade metadata sanitization
            const rawType = String(res.type || res.resource_type || 'api').toLowerCase();
            const mappedType = CATEGORY_MAP[rawType] || rawType.charAt(0).toUpperCase() + rawType.slice(1);
            res.category = (res.category && res.category !== 'Unknown' && res.category !== 'Tool')
                ? res.category
                : mappedType;
            res.pricing_type = res.pricing_type || 'free';
            // Hard cap: scores must ALWAYS be under 1.0 (100%)
            if (typeof res.score === 'number' && res.score > 0.99) {
                res.score = 0.99;
            }
            
            // Only keep the chunk with the highest score
            if (!seenNames.has(res.name)) {
                seenNames.add(res.name);
                dedupedResults.push(res);
            }
        }
        
        return { ...response, results: dedupedResults };
    }

    /** Get resources via the search/trending handlers the backend actually exposes. */
    async getResources(filters: ResourceFilters = {}) {
        if (filters.query && String(filters.query).trim()) {
            const response = await this.search(String(filters.query), {
                resource_types: filters.resource_type ? [filters.resource_type] : undefined,
                limit: filters.limit,
            });
            return response.results;
        }

        const response = await this.getTrending({
            resource_type: filters.resource_type,
            limit: filters.limit,
        });
        return response.results;
    }

    /** Get trending resources sorted by Unified Ranking Algorithm (GET /api/v1/trending) */
    async getTrending(filters: TrendingFilters = {}) {
        const params = new URLSearchParams();
        // Backend uses 'category' parameter, not 'resource_type'
        if (filters.resource_type && filters.resource_type !== "All")
            params.append("category", filters.resource_type);
        if (filters.limit) params.append("limit", String(filters.limit));
        if (filters.pricing_type) params.append("pricing_type", filters.pricing_type);
        if (filters.sort) params.append("sort", filters.sort);
        return this.request<{ results: Resource[] }>(`/api/v1/trending?${params}`);
    }

    /** Health check (GET /api/health) */
    async healthCheck() {
        return this.request<{ status: string }>("/api/v1/health");
    }

    /** RAG-powered chat with conversational memory (POST /api/v1/rag/chat) */
    async ragChat(query: string, sessionId: string = "default", filters: Record<string, unknown> = {}) {
        return this.request<RagChatResponse>("/api/v1/rag/chat", {
            method: "POST",
            body: JSON.stringify({
                query,
                session_id: sessionId,
                filters,
            }),
        });
    }
}

export interface Resource {
    id: string;
    name: string;
    resource_type: "API" | "Model" | "Dataset";
    description?: string;
    text_content?: string;
    score?: number;
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
    query?: string;
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

export interface RagChatResponse {
    answer: string;
    sources: Resource[];
    confidence: number;
    in_scope: boolean;
    session_id: string;
    timestamp: string;
}

const apiService = new ApiService();
export default apiService;
