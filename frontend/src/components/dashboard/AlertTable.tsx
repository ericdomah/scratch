import React, { useState, useEffect } from 'react';
import { Search, Filter, ShieldCheck, AlertTriangle, Activity, Download } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

import { useGridStore } from '../../store/gridStore';

export default function AlertTable() {
  const [searchTerm, setSearchTerm] = useState('');
  const { liveAlerts, addLiveAlert, incrementInvestigations, triggerInspect } = useGridStore();
  const navigate = useNavigate();

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      data.timestamp = new Date().toISOString();
      addLiveAlert(data);
    };
    return () => ws.close();
  }, [addLiveAlert]);

  const handleInspect = (meterId: string) => {
    triggerInspect(meterId);
    incrementInvestigations();
    navigate('/map'); 
  };

  const exportReport = () => {
    const doc = new jsPDF();
    
    // Header
    doc.setFontSize(20);
    doc.setTextColor(0, 0, 0);
    doc.text("KIB-TEK Grid Security Report", 14, 22);
    
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Official System Audit | TRNC Island-Wide Deployment | 1,500 Meters`, 14, 30);
    doc.text(`Generated: ${new Date().toLocaleString()} | Source: GridGuard AI Meta-Ensemble`, 14, 35);
    
    // Anomaly Table
    const tableData = liveAlerts.map(a => [
      a.id,
      a.risk.toUpperCase(),
      `${(a.confidence * 100).toFixed(1)}%`,
      a.status.toUpperCase(),
      new Date(a.timestamp).toLocaleString()
    ]);
    
    autoTable(doc, {
      startY: 45,
      head: [['Meter ID', 'Risk Level', 'AI Confidence', 'Current Status', 'Detection Time']],
      body: tableData,
      headStyles: { fillColor: [15, 23, 42], textColor: [255, 255, 255] },
      alternateRowStyles: { fillColor: [241, 245, 249] },
      margin: { top: 45 },
    });
    
    // High-Priority Data-URI Stream to force filename persistence
    const dataUri = doc.output('datauristring');
    const link = document.createElement('a');
    link.href = dataUri;
    link.download = `KIB_TEK_Anomaly_Report_${new Date().toISOString().split('T')[0]}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-[#050505] h-full flex flex-col text-xs font-mono relative">
      {/* Toolbar */}
      <div className="p-3 border-b border-[#1e293b] bg-[#0a0a0a] flex justify-between items-center gap-4">
        <h3 className="text-lg font-semibold text-white tracking-tight">Active Anomalies</h3>
        <div className="flex space-x-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input 
              type="text" 
              placeholder="Search Meter ID..."
              className="w-full sm:w-64 pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 transition-all font-mono"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <button onClick={exportReport} className="flex items-center px-4 py-2 bg-[#00f0ff] hover:bg-[#00f0ff]/80 border border-[#00f0ff] rounded-none text-sm text-black transition-colors font-bold shadow-lg shadow-[#00f0ff]/20 uppercase tracking-wider">
            <Download className="h-4 w-4 mr-2" />
            Generate Report
          </button>
          <button className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-none text-sm text-slate-300 transition-colors font-medium">
            <Filter className="h-4 w-4 mr-2 text-slate-400" />
            Filter
          </button>
        </div>
      </div>

      {/* Modern Subdued Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-[#0b1121] text-xs uppercase font-bold text-slate-500 border-b border-slate-800 tracking-wider">
            <tr>
              <th className="px-6 py-4 whitespace-nowrap">Meter ID</th>
              <th className="px-6 py-4 whitespace-nowrap">Risk Level</th>
              <th className="px-6 py-4 whitespace-nowrap">AI Confidence</th>
              <th className="px-6 py-4 whitespace-nowrap">Status</th>
              <th className="px-6 py-4 whitespace-nowrap">Detection Time</th>
              <th className="px-6 py-4 text-right whitespace-nowrap">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {liveAlerts.filter(a => a.id.toLowerCase().includes(searchTerm.toLowerCase())).map((alert) => (
              <tr 
                key={alert.id} 
                className="hover:bg-slate-800/40 transition-colors cursor-pointer group"
                onClick={() => handleInspect(alert.id)}
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="font-mono font-medium text-slate-200 group-hover:text-blue-400 transition-colors">
                    {alert.id}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${
                    alert.risk === 'high' 
                      ? 'bg-red-500/10 border-red-500/20 text-red-500' 
                      : 'bg-amber-500/10 border-amber-500/20 text-amber-500'
                  }`}>
                    {alert.risk === 'high' && <span className="w-1.5 h-1.5 rounded-full bg-red-500 mr-1.5 animate-pulse"></span>}
                    {alert.risk.toUpperCase()}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap font-mono text-slate-300">
                  <div className="flex items-center">
                    <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden mr-3">
                      <div 
                        className={`h-full ${alert.confidence > 0.8 ? 'bg-red-500' : 'bg-amber-500'}`} 
                        style={{ width: `${alert.confidence * 100}%` }}
                      ></div>
                    </div>
                    {(alert.confidence * 100).toFixed(1)}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center font-medium">
                    {alert.status === 'investigating' && <Activity className="h-4 w-4 text-amber-500 mr-2" />}
                    {alert.status === 'pending' && <AlertTriangle className="h-4 w-4 text-red-500 mr-2" />}
                    <span className="capitalize">{alert.status}</span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-slate-400 text-xs">
                  {new Date(alert.timestamp).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleInspect(alert.id); }}
                    className="text-blue-500 hover:text-blue-400 font-bold text-xs uppercase tracking-wider bg-blue-500/10 hover:bg-blue-500/20 px-3 py-1.5 rounded transition-colors"
                  >
                    Inspect
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
