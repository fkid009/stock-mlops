'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { format } from 'date-fns';

export default function ModelsPage() {
  const { data: modelsData, isLoading: modelsLoading } = useQuery({
    queryKey: ['models'],
    queryFn: () => api.getModels(),
  });

  const { data: compareData, isLoading: compareLoading } = useQuery({
    queryKey: ['models-compare'],
    queryFn: () => api.compareModels(),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Models</h1>
        <p className="text-gray-600 mt-2">View trained models and performance comparison</p>
      </div>

      {/* Best Model Card */}
      {modelsData?.best_model && (
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg shadow p-6 text-white">
          <h2 className="text-lg font-semibold mb-2">Current Best Model</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-blue-100 text-sm">Model Type</p>
              <p className="text-xl font-bold">{modelsData.best_model.model_type}</p>
            </div>
            <div>
              <p className="text-blue-100 text-sm">Accuracy</p>
              <p className="text-xl font-bold">
                {(modelsData.best_model.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-blue-100 text-sm">Score</p>
              <p className="text-xl font-bold">{modelsData.best_model.score.toFixed(3)}</p>
            </div>
            <div>
              <p className="text-blue-100 text-sm">Trained At</p>
              <p className="text-xl font-bold">
                {modelsData.best_model.trained_at
                  ? format(new Date(modelsData.best_model.trained_at), 'MM/dd HH:mm')
                  : '-'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Model Comparison */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Model Type Comparison</h2>
        {compareLoading ? (
          <div className="animate-pulse space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-gray-200 rounded"></div>
            ))}
          </div>
        ) : compareData?.comparison?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Model Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Avg Accuracy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Max Accuracy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Avg Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Training Count
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Times Best
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {compareData.comparison.map((model: any) => (
                  <tr key={model.model_type} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap font-medium">
                      {model.model_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {(model.avg_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {(model.max_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {model.avg_score.toFixed(3)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">{model.training_count}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{model.times_selected_best}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">No comparison data available</p>
        )}
      </div>

      {/* Training History */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Training History</h2>
        {modelsLoading ? (
          <div className="animate-pulse space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-12 bg-gray-200 rounded"></div>
            ))}
          </div>
        ) : modelsData?.models?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Model Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Accuracy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Best
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Trained At
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {modelsData.models.map((model: any, idx: number) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap font-medium text-sm">
                      {model.model_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{model.model_type}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {(model.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {model.score.toFixed(3)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {model.is_best ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                          Best
                        </span>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {model.trained_at
                        ? format(new Date(model.trained_at), 'yyyy-MM-dd HH:mm')
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">No training history available</p>
        )}
      </div>
    </div>
  );
}
