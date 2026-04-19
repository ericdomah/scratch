import React from 'react';
import { Activity, ShieldAlert, Users, Zap } from 'lucide-react';

export default function KPICards() {
  const kpis = [
    { title: 'Total Meters', value: '1,500', icon: Users, color: 'text-[#00f0ff]' },
    { title: 'Theft Alerts', value: '77', icon: ShieldAlert, color: 'text-red-500' },
    { title: 'Investigations', value: '26', icon: Activity, color: 'text-amber-500' },
    { title: 'Est. Loss (Monthly)', value: '₺821,500', icon: Zap, color: 'text-emerald-500' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 h-full">
      {kpis.map((kpi) => {
        const Icon = kpi.icon;
        return (
          <div key={kpi.title} className="bg-[#050505] border border-[#1e293b] p-5 relative overflow-hidden group flex flex-col justify-between h-28">
            <div className="flex items-center justify-between relative z-10">
              <h3 className="text-[11px] font-bold text-slate-500 uppercase tracking-widest">{kpi.title}</h3>
              <Icon className={`h-4 w-4 ${kpi.color} opacity-80`} />
            </div>
            <div className="relative z-10 mt-auto">
              <p className={`text-4xl font-light ${kpi.color} tracking-tight`}>{kpi.value}</p>
            </div>
            {/* Edge accent line */}
            <div className={`absolute bottom-0 left-0 h-[2px] w-0 group-hover:w-full transition-all duration-500 ${
              kpi.title.includes('Total') ? 'bg-[#00f0ff]' : 
              kpi.title.includes('Alerts') ? 'bg-red-500' : 
              kpi.title.includes('Investigations') ? 'bg-amber-500' : 'bg-emerald-500'
            }`}></div>
          </div>
        );
      })}
    </div>
  );
}
