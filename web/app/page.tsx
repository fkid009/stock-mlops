'use client';

import { useQuery } from '@tanstack/react-query';
import { MetricsSummary } from '@/components/MetricsSummary';
import { PredictionsList } from '@/components/PredictionsList';
import { AccuracyChart } from '@/components/AccuracyChart';
import { ModelStatus } from '@/components/ModelStatus';
import { api } from '@/lib/api';

export default function Dashboard() {
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics-summary'],
    queryFn: () => api.getMetricsSummary(30),
  });

  const { data: predictions, isLoading: predictionsLoading } = useQuery({
    queryKey: ['latest-predictions'],
    queryFn: () => api.getLatestPredictions(10),
  });

  const { data: bestModel, isLoading: modelLoading } = useQuery({
    queryKey: ['best-model'],
    queryFn: () => api.getBestModel(),
  });

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Stock Prediction Dashboard</h1>
        <p className="text-gray-600 mt-2">Monitor predictions and model performance</p>
      </div>

      {/* Metrics Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricsSummary data={metrics} isLoading={metricsLoading} />
      </div>

      {/* Charts and Model Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <AccuracyChart />
        </div>
        <div>
          <ModelStatus data={bestModel} isLoading={modelLoading} />
        </div>
      </div>

      {/* Recent Predictions */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Recent Predictions</h2>
        <PredictionsList data={predictions} isLoading={predictionsLoading} />
      </div>
    </div>
  );
}
