import React from 'react';
import KPICards from '../components/dashboard/KPICards';
import AlertTable from '../components/dashboard/AlertTable';
import WhatIfSimulator from '../components/dashboard/WhatIfSimulator';
import { Activity, ShieldCheck, Terminal } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="h-full flex flex-col font-mono pb-6">
      <div className="flex items-end justify-between border-b border-[#1e293b] pb-4 mb-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-widest uppercase">Grid Operation Control</h2>
          <p className="text-[11px] text-slate-500 mt-1 uppercase tracking-widest">SYSTEM: NOMINAL | V-ID: 8094.AX</p>
        </div>
        <div className="text-right flex space-x-6">
          <div>
            <p className="text-[10px] text-slate-500">SYNC RATE</p>
            <p className="text-sm font-bold text-[#00f0ff]">12ms</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500">THREAT DEFCON</p>
            <div className="flex items-center justify-end mt-1">
              <span className="relative flex h-2 w-2 mr-2">
                <span className="animate-ping absolute inline-flex h-full w-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 bg-green-500"></span>
              </span>
              <span className="text-xs font-bold text-green-500 tracking-widest">SECURE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Brutalist Bento Grid Architecture */}
      <div className="grid grid-cols-12 grid-rows-[auto_1fr] gap-4 flex-1 h-[calc(100vh-140px)]">
        
        {/* KPI Top Row (Spans all columns) */}
        <div className="col-span-12">
          <KPICards />
        </div>

        {/* Lower Left Block: WhatIf Simulator (Spans 5 columns) */}
        <div className="col-span-12 lg:col-span-5 h-full">
          <WhatIfSimulator />
        </div>

        {/* Lower Right Block: Telemetry Matrix (AlertTable) (Spans 7 cols) */}
        <div className="col-span-12 lg:col-span-7 h-full flex flex-col pt-1">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs uppercase font-bold text-slate-400 flex items-center tracking-widest">
              <Terminal className="h-4 w-4 mr-2 text-[#00f0ff]" />
              Real-Time Security Matrix
            </h3>
            <span className="text-[10px] text-[#00f0ff] uppercase tracking-widest">LIVE DATA FEED</span>
          </div>
          <div className="flex-1 overflow-hidden border border-[#1e293b]">
            <AlertTable />
          </div>
        </div>

      </div>
    </div>
  );
}
