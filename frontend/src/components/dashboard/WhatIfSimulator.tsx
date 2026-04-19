import React, { useState } from 'react';
import { Sliders, Activity, Zap, BrainCircuit } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function WhatIfSimulator() {
  const [voltageDrop, setVoltageDrop] = useState(50);
  const [nightUsage, setNightUsage] = useState(20);
  const [ambientTemp, setAmbientTemp] = useState(30);

  // Deep Learning Predictive Mock Algorithm
  const aiConfidence = Math.max(0, Math.min(100, (voltageDrop * 0.8) + (nightUsage * -0.5) + (ambientTemp * 0.1) + 20));

  // Dynamic Chart Data mapping the slider inputs
  const predictiveData = Array.from({ length: 12 }, (_, i) => ({
    time: `${i * 2}h`,
    expected: 45 + Math.sin(i) * 15,
    projected: (45 + Math.sin(i) * 15) * (1 - (aiConfidence / 150))
  }));

  return (
    <div className="bg-[#050505] border border-[#1e293b] p-5 h-full flex flex-col font-mono relative">
      <div className="flex items-center justify-between mb-6 border-b border-[#1e293b] pb-3">
        <h3 className="text-xs uppercase font-bold text-slate-400 flex items-center tracking-widest">
          <BrainCircuit className="h-4 w-4 mr-2 text-[#00f0ff]" />
          Proactive XAI Simulator
        </h3>
        <span className="text-[10px] bg-[#1e293b] text-slate-400 px-2 py-0.5 rounded-sm">SANDBOX</span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 flex-1">
        {/* Sliders Control Panel */}
        <div className="space-y-5">
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-2 font-bold uppercase">
              <span>Voltage Drop (V)</span>
              <span className="text-[#00f0ff]">{voltageDrop}</span>
            </div>
            <input 
              type="range" min="0" max="100" value={voltageDrop} 
              onChange={(e) => setVoltageDrop(Number(e.target.value))}
              className="w-full h-1 bg-[#1e293b] rounded-none appearance-none cursor-pointer accent-[#00f0ff]"
            />
          </div>
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-2 font-bold uppercase">
              <span>Night Load (kWh)</span>
              <span className="text-amber-400">{nightUsage}</span>
            </div>
            <input 
              type="range" min="0" max="100" value={nightUsage} 
              onChange={(e) => setNightUsage(Number(e.target.value))}
              className="w-full h-1 bg-[#1e293b] rounded-none appearance-none cursor-pointer accent-amber-400"
            />
          </div>
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-2 font-bold uppercase">
              <span>Temp Variance (°C)</span>
              <span className="text-emerald-400">{ambientTemp}</span>
            </div>
            <input 
              type="range" min="0" max="100" value={ambientTemp} 
              onChange={(e) => setAmbientTemp(Number(e.target.value))}
              className="w-full h-1 bg-[#1e293b] rounded-none appearance-none cursor-pointer accent-emerald-400"
            />
          </div>

          <div className="bg-[#0a0a0a] border border-[#1e293b] p-4 mt-6">
            <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Simulated Threat Level</p>
            <div className="flex items-end">
              <span className={`text-4xl font-bold ${aiConfidence > 75 ? 'text-red-500' : aiConfidence > 40 ? 'text-amber-500' : 'text-[#00f0ff]'}`}>
                {aiConfidence.toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500 ml-2 mb-1">PROBABILITY</span>
            </div>
          </div>
        </div>

        {/* Dynamic Graphic Output */}
        <div className="bg-[#0a0a0a] border border-[#1e293b] relative">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={predictiveData} margin={{ top: 20, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorProjected" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={aiConfidence > 75 ? '#ef4444' : '#00f0ff'} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={aiConfidence > 75 ? '#ef4444' : '#00f0ff'} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="time" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#000000', border: '1px solid #1e293b', borderRadius: '0', fontSize: '12px' }} 
                itemStyle={{ color: '#f8fafc' }} 
              />
              <Area type="step" dataKey="expected" stroke="#475569" strokeWidth={1} fillOpacity={0} name="Baseline" />
              <Area type="monotone" dataKey="projected" stroke={aiConfidence > 75 ? '#ef4444' : '#00f0ff'} strokeWidth={2} fill="url(#colorProjected)" name="Simulated" />
            </AreaChart>
          </ResponsiveContainer>
          {/* Scanline overlay for aesthetic */}
          <div className="absolute inset-0 pointer-events-none" style={{ backgroundImage: 'linear-gradient(rgba(0,0,0,0) 50%, rgba(0,0,0,0.2) 50%)', backgroundSize: '100% 4px' }}></div>
        </div>
      </div>
    </div>
  );
}
