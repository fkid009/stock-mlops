'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { AccuracyChart } from '@/components/AccuracyChart';
import { format } from 'date-fns';

export default function MetricsPage() {
  const [days, setDays] = useState<number>(30);

  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: ['metrics-summary', days],
    queryFn: () => api.getMetricsSummary(days),
  });

  const { data: metricsData, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics', days],
    queryFn: () => api.getMetrics(days),
  });

  const { data: driftData } = useQuery({
    queryKey: ['drift-status'],
    queryFn: () => api.getDriftStatus(),
  });

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Metrics</h1>
          <p className="text-gray-600 mt-2">Performance metrics and accuracy tracking</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Period</label>
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
            <option value={60}>Last 60 days</option>
            <option value={90}>Last 90 days</option>
          </select>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {summaryLoading ? (
          [1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
              <div className="h-8 bg-gray-200 rounded w-3/4"></div>
            </div>
          ))
        ) : summaryData?.data_available ? (
          <>
            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Overall Accuracy</p>
              <p
                className={`text-2xl font-bold ${
                  summaryData.overall_accuracy > 0.55 ? 'text-green-600' : 'text-yellow-600'
                }`}
              >
                {(summaryData.overall_accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Total Predictions</p>
              <p className="text-2xl font-bold text-blue-600">
                {summaryData.predictions?.total?.toLocaleString() || '0'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Days Tracked</p>
              <p className="text-2xl font-bold text-purple-600">
                {summaryData.total_days_with_data || 0}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Accuracy Range</p>
              <p className="text-2xl font-bold text-gray-600">
                {(summaryData.accuracy?.min * 100).toFixed(0)}% -{' '}
                {(summaryData.accuracy?.max * 100).toFixed(0)}%
              </p>
            </div>
          </>
        ) : (
          <div className="col-span-4 bg-white rounded-lg shadow p-6 text-center text-gray-500">
            No metrics data available yet. Data will appear after predictions are evaluated.
          </div>
        )}
      </div>

      {/* Drift Status */}
      {driftData && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Drift Status</h2>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span
                className={`w-4 h-4 rounded-full ${
                  driftData.drift_detected ? 'bg-red-500' : 'bg-green-500'
                }`}
              ></span>
              <span className="font-medium">
                {driftData.drift_detected ? 'Drift Detected' : 'No Drift'}
              </span>
            </div>
            {driftData.recent_accuracy && (
              <span className="text-gray-500">
                Recent accuracy: {(driftData.recent_accuracy * 100).toFixed(1)}%
              </span>
            )}
            {driftData.baseline_accuracy && (
              <span className="text-gray-500">
                Baseline: {(driftData.baseline_accuracy * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600 mt-2">{driftData.recommendation}</p>
        </div>
      )}

      {/* Accuracy Chart */}
      <AccuracyChart />

      {/* Daily Metrics Table */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Daily Metrics</h2>
        {metricsLoading ? (
          <div className="animate-pulse space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-gray-200 rounded"></div>
            ))}
          </div>
        ) : metricsData?.metrics?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Accuracy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Correct
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Total
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Model
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {metricsData.metrics.map((metric: any, idx: number) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {format(new Date(metric.date), 'yyyy-MM-dd')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`font-medium ${
                          metric.accuracy >= 0.6
                            ? 'text-green-600'
                            : metric.accuracy >= 0.5
                            ? 'text-yellow-600'
                            : 'text-red-600'
                        }`}
                      >
                        {(metric.accuracy * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {metric.correct_predictions}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {metric.total_predictions}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {metric.model_name}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">
            No daily metrics available yet. Data will appear after predictions are evaluated.
          </p>
        )}
      </div>
    </div>
  );
}
