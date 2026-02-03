'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { format } from 'date-fns';

interface ModelStatusProps {
  data: any;
  isLoading: boolean;
  error?: Error | null;
}

export function ModelStatus({ data, isLoading, error }: ModelStatusProps) {
  const { data: driftData, error: driftError } = useQuery({
    queryKey: ['drift-status'],
    queryFn: () => api.getDriftStatus(),
  });

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Model Status</h3>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Model Status</h3>
        <div className="text-red-600">
          <p className="font-medium">Failed to load model status</p>
          <p className="text-sm mt-1">{error.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Model Status</h3>

      {data ? (
        <div className="space-y-4">
          <div>
            <p className="text-sm text-gray-500">Current Best Model</p>
            <p className="text-lg font-medium text-gray-900">{data.model_type}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Accuracy</p>
            <p className="text-lg font-medium text-green-600">
              {(data.accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Last Trained</p>
            <p className="text-sm text-gray-700">
              {data.trained_at
                ? format(new Date(data.trained_at), 'yyyy-MM-dd HH:mm')
                : 'Unknown'}
            </p>
          </div>
        </div>
      ) : (
        <p className="text-gray-500">No model trained yet</p>
      )}

      {/* Drift Status */}
      <div className="mt-6 pt-6 border-t">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Drift Status</h4>
        {driftData ? (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span
                className={`w-3 h-3 rounded-full ${
                  driftData.drift_detected ? 'bg-red-500' : 'bg-green-500'
                }`}
              ></span>
              <span className="text-sm">
                {driftData.drift_detected ? 'Drift Detected' : 'No Drift'}
              </span>
            </div>
            {driftData.recent_accuracy && (
              <p className="text-xs text-gray-500">
                Recent accuracy: {(driftData.recent_accuracy * 100).toFixed(1)}%
              </p>
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-500">Unable to check drift status</p>
        )}
      </div>
    </div>
  );
}
