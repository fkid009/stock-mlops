import axios from 'axios';

const API_URL = {
  production: 'https://api.lhb99.com',
  development: 'http://localhost:8000',
};

const getApiBase = () => {
  if (typeof window === 'undefined') {
    return process.env.NEXT_PUBLIC_API_URL || API_URL.development;
  }
  if (window.location.hostname.endsWith('lhb99.com')) {
    return API_URL.production;
  }
  return API_URL.development;
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
