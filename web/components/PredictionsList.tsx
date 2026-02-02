'use client';

import { format } from 'date-fns';

interface PredictionsListProps {
  data: any;
  isLoading: boolean;
}

export function PredictionsList({ data, isLoading }: PredictionsListProps) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="animate-pulse p-4 space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex space-x-4">
              <div className="h-4 bg-gray-200 rounded w-1/4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/4"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const predictions = data?.predictions || [];

  if (predictions.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500">
        No predictions available
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Prediction</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Probability</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actual</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Result</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {predictions.map((pred: any, idx: number) => {
            const isCorrect = pred.actual !== null && pred.prediction === pred.actual;
            const isIncorrect = pred.actual !== null && pred.prediction !== pred.actual;

            return (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap font-medium">{pred.symbol}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {format(new Date(pred.date), 'yyyy-MM-dd')}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    pred.prediction === 1
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {pred.prediction === 1 ? 'UP' : 'DOWN'}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  {pred.probability ? `${(pred.probability * 100).toFixed(1)}%` : '-'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {pred.actual !== null ? (
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      pred.actual === 1
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {pred.actual === 1 ? 'UP' : 'DOWN'}
                    </span>
                  ) : (
                    <span className="text-gray-400">Pending</span>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {isCorrect && <span className="text-green-600 font-medium">✓ Correct</span>}
                  {isIncorrect && <span className="text-red-600 font-medium">✗ Wrong</span>}
                  {pred.actual === null && <span className="text-gray-400">-</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
