import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';

const trendData = [
  { name: 'Mon', alerts: 12, fixed: 8 },
  { name: 'Tue', alerts: 19, fixed: 15 },
  { name: 'Wed', alerts: 8, fixed: 20 },
  { name: 'Thu', alerts: 25, fixed: 10 },
  { name: 'Fri', alerts: 14, fixed: 14 },
  { name: 'Sat', alerts: 30, fixed: 5 },
  { name: 'Sun', alerts: 10, fixed: 25 },
];

export default function Analytics() {
  return (
    <div className="space-y-6 pb-12">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">System Analytics</h2>
        <p className="text-sm text-slate-400 mt-1">GridGuard ML performance and historical anomaly frequency routing.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-sm">
          <h3 className="text-sm font-semibold text-slate-300 mb-6 uppercase tracking-wider">7-Day Alert Frequency vs Resolutions</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorAlerts" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorFixed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} axisLine={false} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} axisLine={false} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} 
                  itemStyle={{ color: '#f8fafc' }} 
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Area type="monotone" dataKey="alerts" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorAlerts)" name="New Anomalies" />
                <Area type="monotone" dataKey="fixed" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorFixed)" name="Resolved Confirmed" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-sm">
          <h3 className="text-sm font-semibold text-slate-300 mb-6 uppercase tracking-wider">ML Model Confidence Distribution</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                { range: '50-60%', count: 5 },
                { range: '60-70%', count: 12 },
                { range: '70-80%', count: 34 },
                { range: '80-90%', count: 105 },
                { range: '90-100%', count: 280 },
              ]} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="range" stroke="#64748b" fontSize={12} axisLine={false} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} axisLine={false} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} 
                  cursor={{fill: '#1e293b', opacity: 0.4}} 
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Detections Logged" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
