import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, Legend, LineChart, Line, AreaChart, Area
} from 'recharts';
import { TrendingUp, PieChart as PieIcon, Map as MapIcon, Clock, DollarSign } from 'lucide-react';

const RECOVERY_DATA = [
  { month: 'Jan', amount: 420000 },
  { month: 'Feb', amount: 510000 },
  { month: 'Mar', amount: 480000 },
  { month: 'Apr', amount: 620000 },
  { month: 'May', amount: 750000 },
  { month: 'Jun', amount: 890000 },
];

const LOSS_DECOMPOSITION = [
  { name: 'Technical Loss', value: 35, color: '#64748b' },
  { name: 'Non-Technical (Theft)', value: 65, color: '#f43f5e' },
];

const DISTRICT_RISK = [
  { district: 'Lefkoşa', theftRate: '12.4%', risk: 'High', color: 'text-red-500' },
  { district: 'Girne', theftRate: '8.2%', risk: 'Medium', color: 'text-amber-500' },
  { district: 'Gazimağusa', theftRate: '14.1%', risk: 'Critical', color: 'text-rose-600' },
  { district: 'Güzelyurt', theftRate: '5.4%', risk: 'Low', color: 'text-emerald-500' },
  { district: 'İskele', theftRate: '9.8%', risk: 'Medium', color: 'text-amber-500' },
];

const TEMPORAL_PATTERN = [
  { hour: '00', prob: 20 }, { hour: '02', prob: 85 }, { hour: '04', prob: 95 },
  { hour: '06', prob: 40 }, { hour: '08', prob: 15 }, { hour: '10', prob: 10 },
  { hour: '12', prob: 12 }, { hour: '14', prob: 18 }, { hour: '16', prob: 25 },
  { hour: '18', prob: 30 }, { hour: '20', prob: 45 }, { hour: '22', prob: 60 },
];

export default function ForensicPanel() {
  return (
    <div className="bg-[#050505] border border-[#1e293b] p-6 font-mono">
      <div className="flex items-center justify-between mb-8 border-b border-[#1e293b] pb-4">
        <div>
          <h2 className="text-sm font-black text-white uppercase tracking-[0.2em] flex items-center">
            <TrendingUp className="h-4 w-4 mr-2 text-[#00f0ff]" />
            Grid Financial & Forensic Analytics
          </h2>
          <p className="text-[10px] text-slate-500 mt-1 uppercase">Advanced Recovery Forecasting | TRNC Island-Wide Profile</p>
        </div>
        <div className="flex space-x-4">
          <div className="text-right">
            <p className="text-[9px] text-slate-500 uppercase">Est. Annual Recovery</p>
            <p className="text-lg font-black text-[#00f0ff]">$8.42M</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        
        {/* Revenue Recovery Forecast */}
        <div className="col-span-12 lg:col-span-4 space-y-4">
          <div className="flex items-center space-x-2 mb-2">
            <DollarSign className="h-3 w-3 text-slate-500" />
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Revenue Recovery Forecast</h3>
          </div>
          <div className="h-[200px] w-full bg-[#0a0a0a] border border-[#1e293b] p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={RECOVERY_DATA}>
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
                  data={LOSS_DECOMPOSITION}
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {LOSS_DECOMPOSITION.map((entry, index) => (
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
              <span className="text-xl font-black text-white">18.4%</span>
            </div>
          </div>
        </div>

        {/* District Risk Leaderboard */}
        <div className="col-span-12 lg:col-span-5 space-y-4">
          <div className="flex items-center space-x-2 mb-2">
            <MapIcon className="h-3 w-3 text-slate-500" />
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">District Risk ranking</h3>
          </div>
          <div className="bg-[#0a0a0a] border border-[#1e293b] overflow-hidden">
            <table className="w-full text-left">
              <thead className="bg-[#0f172a] text-[9px] uppercase font-bold text-slate-500 border-b border-[#1e293b]">
                <tr>
                  <th className="px-4 py-2">District</th>
                  <th className="px-4 py-2">Theft Rate</th>
                  <th className="px-4 py-2 text-right">Risk Level</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e293b]">
                {DISTRICT_RISK.map((d, i) => (
                  <tr key={i} className="hover:bg-white/5 transition-colors">
                    <td className="px-4 py-2 text-[10px] font-bold text-slate-300">{d.district}</td>
                    <td className="px-4 py-2 text-[10px] font-mono text-[#00f0ff]">{d.theftRate}</td>
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
