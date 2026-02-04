'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { PredictionsList } from '@/components/PredictionsList';
import { api } from '@/lib/api';

export default function PredictionsPage() {
  const [symbol, setSymbol] = useState<string>('');
  const [limit, setLimit] = useState<number>(50);

  const { data, isLoading, error } = useQuery({
    queryKey: ['predictions', symbol, limit],
    queryFn: () => api.getPredictions({ symbol: symbol || undefined, limit }),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Predictions</h1>
        <p className="text-gray-600 mt-2">View all stock movement predictions</p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Symbol
            </label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Symbols</option>
              <option value="AAPL">AAPL</option>
              <option value="GOOGL">GOOGL</option>
              <option value="MSFT">MSFT</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Limit
            </label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
            </select>
          </div>
        </div>
      </div>

      {/* Predictions Table */}
      <PredictionsList
        data={data}
        isLoading={isLoading}
        error={error as Error | null}
      />
    </div>
  );
}
