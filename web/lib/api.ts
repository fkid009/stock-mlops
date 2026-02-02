import axios from 'axios';

// Use relative URL in browser (will be proxied or same origin)
// For SSR/server-side, use the internal Docker URL
const getApiBase = () => {
  if (typeof window === 'undefined') {
    // Server-side: use Docker internal network
    return process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';
  }
  // Client-side: use relative URL or current host
  // API should be accessible at the same host on port 8000
  const host = window.location.hostname;
  return `http://${host}:8000`;
};

const client = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
});

// Set baseURL dynamically for each request
client.interceptors.request.use((config) => {
  config.baseURL = getApiBase();
  return config;
});

export const api = {
  // Predictions
  async getLatestPredictions(limit: number = 10) {
    const { data } = await client.get(`/api/predictions?limit=${limit}`);
    return data;
  },

  async getPredictions(params?: {
    symbol?: string;
    startDate?: string;
    endDate?: string;
    limit?: number;
  }) {
    const { data } = await client.get('/api/predictions', { params });
    return data;
  },

  async getPredictionsBySymbol(symbol: string) {
    const { data } = await client.get(`/api/predictions?symbol=${symbol}`);
    return data;
  },

  // Models
  async getModels() {
    const { data } = await client.get('/api/models');
    return data;
  },

  async getBestModel() {
    const { data } = await client.get('/api/models/best');
    return data;
  },

  async compareModels() {
    const { data } = await client.get('/api/models/compare');
    return data;
  },

  // Metrics
  async getMetrics(days: number = 30) {
    const { data } = await client.get(`/api/metrics?days=${days}`);
    return data;
  },

  async getMetricsSummary(days: number = 30) {
    const { data } = await client.get(`/api/metrics/summary?days=${days}`);
    return data;
  },

  async getAccuracyTrend(days: number = 30, window: number = 5) {
    const { data } = await client.get(`/api/metrics/trend?days=${days}&window=${window}`);
    return data;
  },

  async getDriftStatus() {
    const { data } = await client.get('/api/metrics/drift');
    return data;
  },
};
