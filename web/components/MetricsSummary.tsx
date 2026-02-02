'use client';

interface MetricsSummaryProps {
  data: any;
  isLoading: boolean;
}

export function MetricsSummary({ data, isLoading }: MetricsSummaryProps) {
  if (isLoading) {
    return (
      <>
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-3/4"></div>
          </div>
        ))}
      </>
    );
  }

  if (!data || !data.data_available) {
    return (
      <div className="col-span-4 bg-white rounded-lg shadow p-6 text-center text-gray-500">
        No metrics data available
      </div>
    );
  }

  const cards = [
    {
      label: 'Overall Accuracy',
      value: `${(data.overall_accuracy * 100).toFixed(1)}%`,
      color: data.overall_accuracy > 0.55 ? 'text-green-600' : 'text-yellow-600',
    },
    {
      label: 'Total Predictions',
      value: data.predictions?.total?.toLocaleString() || '0',
      color: 'text-blue-600',
    },
    {
      label: 'Days Tracked',
      value: data.total_days_with_data || 0,
      color: 'text-purple-600',
    },
    {
      label: 'Accuracy Range',
      value: `${(data.accuracy?.min * 100).toFixed(0)}% - ${(data.accuracy?.max * 100).toFixed(0)}%`,
      color: 'text-gray-600',
    },
  ];

  return (
    <>
      {cards.map((card) => (
        <div key={card.label} className="bg-white rounded-lg shadow p-6">
          <p className="text-sm text-gray-500">{card.label}</p>
          <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
        </div>
      ))}
    </>
  );
}
