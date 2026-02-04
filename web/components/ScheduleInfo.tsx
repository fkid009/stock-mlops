'use client';

export function ScheduleInfo() {
  return (
    <div className="bg-gray-50 border-t border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-xs">
          {/* NYSE/NASDAQ */}
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-green-400 rounded-full"></span>
            <span className="font-medium text-gray-700">NYSE/NASDAQ</span>
            <div className="flex items-center gap-1.5 text-gray-500">
              <span>09:30-16:00 <span className="text-gray-400">EST</span></span>
              <span className="text-gray-300">→</span>
              <span>23:30-익일 06:00 <span className="text-gray-400">KST</span></span>
            </div>
          </div>

          {/* Daily Training */}
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
            <span className="font-medium text-gray-700">Daily Training</span>
            <div className="flex items-center gap-1.5 text-gray-500">
              <span>평일 17:00 <span className="text-gray-400">EST</span></span>
              <span className="text-gray-300">→</span>
              <span>익일 07:00 <span className="text-gray-400">KST</span></span>
            </div>
          </div>

          {/* Weekly Tuning */}
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
            <span className="font-medium text-gray-700">Weekly Tuning</span>
            <div className="flex items-center gap-1.5 text-gray-500">
              <span>일요일 17:00 <span className="text-gray-400">EST</span></span>
              <span className="text-gray-300">→</span>
              <span>월요일 07:00 <span className="text-gray-400">KST</span></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
