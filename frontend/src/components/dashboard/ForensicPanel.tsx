import React, { useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, LineChart, Line, AreaChart, Area
} from 'recharts';
import { TrendingUp, PieChart as PieIcon, Map as MapIcon, Clock, DollarSign } from 'lucide-react';
import { useGridStore } from '../../store/gridStore';

const TEMPORAL_PATTERN = [
  { hour: '00', prob: 20 }, { hour: '02', prob: 85 }, { hour: '04', prob: 95 },
  { hour: '06', prob: 40 }, { hour: '08', prob: 15 }, { hour: '10', prob: 10 },
  { hour: '12', prob: 12 }, { hour: '14', prob: 18 }, { hour: '16', prob: 25 },
  { hour: '18', prob: 30 }, { hour: '20', prob: 45 }, { hour: '22', prob: 60 },
];

export default function ForensicPanel() {
  const { liveAlerts } = useGridStore();

  // Dynamic Calculation: District Risk
  const districtRanking = useMemo(() => {
    const districts: Record<string, number> = {};
    liveAlerts.forEach(alert => {
      const d = (alert as any).district || 'Unknown';
      districts[d] = (districts[d] || 0) + 1;
    });

    return Object.entries(districts)
      .map(([name, count]) => ({
        name,
        count,
        risk: count > 10 ? 'Critical' : count > 5 ? 'High' : 'Medium',
        color: count > 10 ? 'text-rose-600' : count > 5 ? 'text-red-500' : 'text-amber-500'
      }))
      .sort((a, b) => b.count - a.count);
  }, [liveAlerts]);

  // Dynamic Calculation: Loss Decomposition
  const totalMeters = 1500;
  const highRiskCount = liveAlerts.filter(a => a.risk === 'high').length;
  const theftLossPercent = ((highRiskCount / totalMeters) * 100).toFixed(1);
  const technicalLossPercent = 5.2; // Fixed baseline technical loss for TRNC grid

  const lossData = [
    { name: 'Technical Loss', value: parseFloat(technicalLossPercent.toString()), color: '#64748b' },
    { name: 'Non-Technical (Theft)', value: parseFloat(theftLossPercent), color: '#f43f5e' },
  ];

  // Dynamic Calculation: Annual Recovery Forecast
  const estRecovery = (highRiskCount * 4200 * 1.5).toLocaleString(); // $4200 avg recovery per theft unit

  const recoveryChartData = [
    { month: 'Jan', amount: highRiskCount * 450 },
    { month: 'Feb', amount: highRiskCount * 510 },
    { month: 'Mar', amount: highRiskCount * 480 },
    { month: 'Apr', amount: highRiskCount * 620 },
    { month: 'May', amount: highRiskCount * 750 },
    { month: 'Jun', amount: highRiskCount * 890 },
  ];

  return (
    <div className="bg-[#050505] border border-[#1e293b] p-6 font-mono">
      <div className="flex items-center justify-between mb-8 border-b border-[#1e293b] pb-4">
        <div>
          <h2 className="text-sm font-black text-white uppercase tracking-[0.2em] flex items-center">
            <TrendingUp className="h-4 w-4 mr-2 text-[#00f0ff]" />
            Grid Financial & Forensic Analytics (LIVE)
          </h2>
          <p className="text-[10px] text-slate-500 mt-1 uppercase">Reactive Recovery Forecasting | Synced to AI Detection Engine</p>
        </div>
        <div className="flex space-x-4">
          <div className="text-right">
            <p className="text-[9px] text-slate-500 uppercase">Est. Annual Recovery</p>
            <p className="text-lg font-black text-[#00f0ff]">${estRecovery}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        
        {/* Revenue Recovery Forecast */}
        <div className="col-span-12 lg:col-span-4 space-y-4">
          <div className="flex items-center space-x-2 mb-2">
            <DollarSign className="h-3 w-3 text-slate-500" />
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Recovery Forecast ($)</h3>
          </div>
          <div className="h-[200px] w-full bg-[#0a0a0a] border border-[#1e293b] p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={recoveryChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="month" stroke="#475569" fontSize={9} tickLine={false} axisLine={false} />
                <YAxis hide />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#050505', border: '1px solid #1e293b', fontSize: '10px' }}
                  cursor={{ fill: '#1e293b' }}
                />
                <Bar dataKey="amount" fill="#00f0ff" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Grid Loss Decomposition */}
        <div className="col-span-12 lg:col-span-3 space-y-4">
          <div className="flex items-center space-x-2 mb-2">
            <PieIcon className="h-3 w-3 text-slate-500" />
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Loss Decomposition</h3>
          </div>
          <div className="h-[200px] w-full bg-[#0a0a0a] border border-[#1e293b] flex items-center justify-center relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={lossData}
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {lossData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#050505', border: '1px solid #1e293b', fontSize: '10px' }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-[10px] text-slate-500 uppercase font-bold">Total Loss</span>
              <span className="text-xl font-black text-white">{(parseFloat(theftLossPercent) + technicalLossPercent).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* District Risk Leaderboard */}
        <div className="col-span-12 lg:col-span-5 space-y-4">
          <div className="flex items-center space-x-2 mb-2">
            <MapIcon className="h-3 w-3 text-slate-500" />
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Live District Risk Ranking</h3>
          </div>
          <div className="bg-[#0a0a0a] border border-[#1e293b] overflow-hidden">
            <table className="w-full text-left">
              <thead className="bg-[#0f172a] text-[9px] uppercase font-bold text-slate-500 border-b border-[#1e293b]">
                <tr>
                  <th className="px-4 py-2">District</th>
                  <th className="px-4 py-2">Active Alerts</th>
                  <th className="px-4 py-2 text-right">Risk Level</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e293b]">
                {districtRanking.map((d, i) => (
                  <tr key={i} className="hover:bg-white/5 transition-colors">
                    <td className="px-4 py-2 text-[10px] font-bold text-slate-300">{d.name}</td>
                    <td className="px-4 py-2 text-[10px] font-mono text-[#00f0ff]">{d.count}</td>
                    <td className={`px-4 py-2 text-[9px] font-black uppercase text-right ${d.color}`}>{d.risk}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Temporal Theft Pattern */}
        <div className="col-span-12 space-y-4 pt-4 border-t border-[#1e293b]/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Clock className="h-3 w-3 text-slate-500" />
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Temporal Theft Probability Profile</h3>
            </div>
            <p className="text-[9px] text-rose-500 font-bold uppercase animate-pulse">Critical: Peak Theft activity 02:00 - 05:00</p>
          </div>
          <div className="h-[120px] w-full bg-[#0a0a0a] border border-[#1e293b] p-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={TEMPORAL_PATTERN}>
                <defs>
                  <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="hour" stroke="#475569" fontSize={9} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#050505', border: '1px solid #1e293b', fontSize: '10px' }}
                />
                <Area type="monotone" dataKey="prob" stroke="#f43f5e" fillOpacity={1} fill="url(#colorProb)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}

