import React from 'react';
import KPICards from '../components/dashboard/KPICards';
import AlertTable from '../components/dashboard/AlertTable';
import WhatIfSimulator from '../components/dashboard/WhatIfSimulator';
import GeospatialMap from './GeospatialMap';
import { Activity, ShieldCheck, Terminal, MapPin } from 'lucide-react';

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

      {/* Brutalist Bento Grid Architecture (Now scrollable for Map) */}
      <div className="grid grid-cols-12 gap-5 flex-1 overflow-y-auto pb-10 pr-2 custom-scrollbar">
        
        {/* KPI Top Row (Spans all columns) */}
        <div className="col-span-12">
          <KPICards />
        </div>

        {/* Central Map Array (Spans 8 columns) */}
        <div className="col-span-12 xl:col-span-8 h-[550px] flex flex-col">
          <div className="flex items-center justify-between mb-3 border-b border-[#1e293b] pb-2">
            <h3 className="text-xs uppercase font-bold text-slate-400 flex items-center tracking-widest">
              <MapPin className="h-4 w-4 mr-2 text-rose-500" />
              Live TRNC Substation Map
            </h3>
            <span className="text-[10px] bg-[#1e293b] text-slate-400 px-2 py-0.5 rounded-sm">KIB-TEK TOPOLOGY</span>
          </div>
          <div className="flex-1 rounded-none border border-[#1e293b]">
            <GeospatialMap />
          </div>
        </div>

        {/* Right Block: WhatIf Simulator (Spans 4 columns) */}
        <div className="col-span-12 xl:col-span-4 h-[550px] flex flex-col">
          <WhatIfSimulator />
        </div>

        {/* Lower Row: Telemetry Matrix (AlertTable) (Spans all cols) */}
        <div className="col-span-12 h-[600px] flex flex-col pt-4">
          <div className="flex items-center justify-between mb-3 border-b border-[#1e293b] pb-2">
            <h3 className="text-xs uppercase font-bold text-slate-400 flex items-center tracking-widest">
              <Terminal className="h-4 w-4 mr-2 text-[#00f0ff]" />
              Real-Time Security Matrix
            </h3>
            <span className="text-[10px] text-[#00f0ff] uppercase tracking-widest animate-pulse">LIVE DATA FEED</span>
          </div>
          <div className="flex-1 overflow-hidden border border-[#1e293b] rounded-none">
            <AlertTable />
          </div>
        </div>

      </div>
    </div>
  );
}
