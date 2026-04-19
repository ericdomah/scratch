import React, { useState } from 'react';
import { Search, Filter, Download, ExternalLink, ShieldAlert, CheckCircle } from 'lucide-react';
import { useGridStore } from '../store/gridStore';
import { useNavigate } from 'react-router-dom';

export default function Alerts() {
  const { liveAlerts, setSelectedMeterId } = useGridStore();
  const [filter, setFilter] = useState('all');
  const navigate = useNavigate();

  const filteredAlerts = liveAlerts.filter(a => {
    if (filter === 'all') return true;
    return a.risk === filter;
  });

  const handleInspect = (id: string) => {
    setSelectedMeterId(id);
    navigate('/map');
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight font-mono">Central Alert Matrix</h2>
          <p className="text-sm text-slate-400 mt-1">Real-time anomaly stream from TRNC smart meter infrastructure.</p>
        </div>
        <div className="flex space-x-3">
          <div className="flex bg-[#050505] border border-slate-800 p-1">
            <button 
              onClick={() => setFilter('all')}
              className={`px-3 py-1 text-[10px] font-bold uppercase transition-all ${filter === 'all' ? 'bg-[#00f0ff] text-black' : 'text-slate-500 hover:text-white'}`}
            >
              All
            </button>
            <button 
              onClick={() => setFilter('high')}
              className={`px-3 py-1 text-[10px] font-bold uppercase transition-all ${filter === 'high' ? 'bg-red-500 text-white' : 'text-slate-500 hover:text-white'}`}
            >
              High Risk
            </button>
            <button 
              onClick={() => setFilter('medium')}
              className={`px-3 py-1 text-[10px] font-bold uppercase transition-all ${filter === 'medium' ? 'bg-amber-500 text-white' : 'text-slate-500 hover:text-white'}`}
            >
              Medium
            </button>
          </div>
        </div>
      </div>

      <div className="bg-[#050505] border border-slate-800 rounded-none overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-[#0a0a0a]">
              <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Meter ID</th>
              <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Risk Level</th>
              <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Confidence</th>
              <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Timestamp</th>
              <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {filteredAlerts.map((alert) => (
              <tr key={alert.id} className="hover:bg-slate-900/40 transition-colors group">
                <td className="px-6 py-4">
                  <span className="text-sm font-bold text-white font-mono">{alert.id}</span>
                </td>
                <td className="px-6 py-4">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                    alert.risk === 'high' ? 'bg-red-500/10 text-red-500' : 'bg-amber-500/10 text-amber-500'
                  }`}>
                    <ShieldAlert className="h-3 w-3 mr-1" />
                    {alert.risk}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 h-1 bg-slate-800 w-16 overflow-hidden">
                      <div 
                        className={`h-full ${alert.risk === 'high' ? 'bg-red-500' : 'bg-amber-500'}`} 
                        style={{ width: `${alert.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-[11px] text-slate-400 font-mono">{(alert.confidence * 100).toFixed(1)}%</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="text-xs text-slate-500 font-mono">{new Date(alert.timestamp).toLocaleString()}</span>
                </td>
                <td className="px-6 py-4 text-right">
                  <button 
                    onClick={() => handleInspect(alert.id)}
                    className="text-slate-400 hover:text-[#00f0ff] p-1 transition-colors"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        
        {filteredAlerts.length === 0 && (
          <div className="py-20 text-center text-slate-500 uppercase text-xs tracking-widest font-bold">
            Zero anomalies detected in current sector.
          </div>
        )}
      </div>
    </div>
  );
}
