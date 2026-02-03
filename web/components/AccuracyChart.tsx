'use client';

import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { api } from '@/lib/api';
import { format } from 'date-fns';

export function AccuracyChart() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['accuracy-trend'],
    queryFn: () => api.getAccuracyTrend(30, 5),
  });

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Accuracy Trend</h3>
        <div className="h-64 flex items-center justify-center">
          <div className="animate-pulse text-gray-400">Loading chart...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Accuracy Trend</h3>
        <div className="h-64 flex flex-col items-center justify-center">
          <p className="text-red-600 font-medium">Failed to load chart data</p>
          <p className="text-red-500 text-sm mt-1">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!data?.trend?.length) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Accuracy Trend</h3>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No trend data available yet. Data will appear after predictions are evaluated.
        </div>
      </div>
    );
  }

  const chartData = data.trend.map((item: any) => ({
    ...item,
    date: format(new Date(item.date), 'MM/dd'),
    accuracy: (item.accuracy * 100).toFixed(1),
    accuracy_ma: (item.accuracy_ma * 100).toFixed(1),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Accuracy Trend (Last 30 Days)</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" fontSize={12} />
            <YAxis domain={[0, 100]} fontSize={12} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              formatter={(value: any) => [`${value}%`, '']}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#9CA3AF"
              strokeWidth={1}
              dot={false}
              name="Daily"
            />
            <Line
              type="monotone"
              dataKey="accuracy_ma"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
              name="5-Day MA"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
