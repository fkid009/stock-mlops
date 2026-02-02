import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Predictions
  async getLatestPredictions(limit: number = 10) {
    const { data } = await client.get(`/predictions?limit=${limit}`);
    return data;
  },

  async getPredictions(params?: {
    symbol?: string;
    startDate?: string;
    endDate?: string;
    limit?: number;
  }) {
    const { data } = await client.get('/predictions', { params });
    return data;
  },

  async getPredictionsBySymbol(symbol: string) {
    const { data } = await client.get(`/predictions?symbol=${symbol}`);
    return data;
  },

  // Models
  async getModels() {
    const { data } = await client.get('/models');
    return data;
  },

  async getBestModel() {
    const { data } = await client.get('/models/best');
    return data;
  },

  async compareModels() {
    const { data } = await client.get('/models/compare');
    return data;
  },

  // Metrics
  async getMetrics(days: number = 30) {
    const { data } = await client.get(`/metrics?days=${days}`);
    return data;
  },

  async getMetricsSummary(days: number = 30) {
    const { data } = await client.get(`/metrics/summary?days=${days}`);
    return data;
  },

  async getAccuracyTrend(days: number = 30, window: number = 5) {
    const { data } = await client.get(`/metrics/trend?days=${days}&window=${window}`);
    return data;
  },

  async getDriftStatus() {
    const { data } = await client.get('/metrics/drift');
    return data;
  },
};
