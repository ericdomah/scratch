import React, { useState } from 'react';
import { Search, Filter, Clock, CheckCircle, AlertTriangle, ChevronRight, MapPin, Download } from 'lucide-react';
import { useGridStore } from '../store/gridStore';
import { useNavigate } from 'react-router-dom';
import jsPDF from 'jspdf';

export default function Investigations() {
  const { liveAlerts, triggerInspect } = useGridStore();
  const [selectedCase, setSelectedCase] = useState<any>(null);
  const [currentTime, setCurrentTime] = useState(Date.now());
  const navigate = useNavigate();

  const generatePDF = () => {
    if (!selectedCase) return;
    const doc = new jsPDF();
    doc.setFontSize(22);
    doc.text("GridGuard Forensic Audit Report", 14, 22);
    doc.setFontSize(12);
    doc.text(`Case ID: ${selectedCase.id}`, 14, 34);
    doc.text(`Risk Level: ${selectedCase.risk}`, 14, 42);
    doc.text(`Detection Time: ${new Date(selectedCase.timestamp).toLocaleString()}`, 14, 50);
    
    doc.setFontSize(14);
    doc.text("1. Forensic Summary", 14, 65);
    doc.setFontSize(11);
    doc.text("The Deep Learning Meta-Ensemble detected a high-confidence theft signature", 14, 73);
    doc.text("correlating with a partial meter bypass or phase tampering.", 14, 79);
    
    doc.setFontSize(14);
    doc.text("2. Time-Series XAI Heatmap (Integrated Gradients)", 14, 95);
    doc.setFontSize(11);
    doc.text("The attached heatmap shows the exact temporal sequence where the anomaly occurred.", 14, 103);
    
    // Placeholder rect for where image would be embedded in full prod
    doc.setDrawColor(0);
    doc.setFillColor(245, 245, 245);
    doc.rect(14, 110, 180, 80, 'FD');
    doc.text("XAI Heatmap Output Rendered Below", 70, 150);
    
    doc.save(`GridGuard_Audit_${selectedCase.id}.pdf`);
  };
  
  React.useEffect(() => {
    const timer = setInterval(() => setCurrentTime(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Filter for items that are being investigated
  const investigations = liveAlerts.filter(a => a.status === 'investigating');

  const formatTimeActive = (timestamp: string) => {
    const diff = Math.floor((currentTime - new Date(timestamp).getTime()) / 1000);
    if (diff < 0) return '0s';
    const hours = Math.floor(diff / 3600);
    const mins = Math.floor((diff % 3600) / 60);
    const secs = diff % 60;
    return `${hours}h ${mins}m ${secs}s`;
  };

  return (
    <div className="space-y-6 pb-12 relative min-h-[600px]">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Active Investigations</h2>
          <p className="text-sm text-slate-400 mt-1">Field operations and forensic analysis tracking for TRNC grid security.</p>
        </div>
        <div className="flex space-x-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input 
              type="text" 
              placeholder="Search Case ID..."
              className="pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500 transition-all"
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {investigations.length === 0 ? (
            <div className="bg-slate-900/50 border border-slate-800 border-dashed rounded-xl p-12 text-center">
              <CheckCircle className="h-12 w-12 text-slate-700 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-400">No active investigations</h3>
              <p className="text-slate-500 text-sm mt-1">Assign an anomaly from the dashboard to begin a field audit.</p>
            </div>
          ) : (
            investigations.map((issue) => (
              <div 
                key={issue.id} 
                onClick={() => setSelectedCase(issue)}
                className={`bg-slate-900 border rounded-xl p-4 transition-all group cursor-pointer ${selectedCase?.id === issue.id ? 'border-[#00f0ff] ring-1 ring-[#00f0ff]' : 'border-slate-800 hover:border-slate-700'}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className={`p-3 rounded-lg ${issue.risk === 'high' ? 'bg-red-500/10 text-red-500' : 'bg-amber-500/10 text-amber-500'}`}>
                      <AlertTriangle className="h-6 w-6" />
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <h4 className="font-bold text-white uppercase tracking-tight">{issue.id}</h4>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${issue.risk === 'high' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
                          {issue.risk} RISK
                        </span>
                      </div>
                      <div className="flex items-center space-x-3 mt-1 text-xs text-slate-500">
                        <span className="flex items-center"><Clock className="h-3 w-3 mr-1" /> Opened {new Date(issue.timestamp).toLocaleTimeString()}</span>
                        <span>Assigned to: Field Team Alpha</span>
                      </div>
                    </div>
                  </div>
                  <ChevronRight className={`h-5 w-5 transition-colors ${selectedCase?.id === issue.id ? 'text-[#00f0ff]' : 'text-slate-700 group-hover:text-white'}`} />
                </div>
              </div>
            ))
          )}
        </div>

        {/* Forensic Detail Panel */}
        <div className="lg:col-span-1">
          {selectedCase ? (
            <div className="bg-[#050505] border border-slate-800 p-6 rounded-xl sticky top-24 animate-in fade-in slide-in-from-right-4">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Investigation Details</h3>
              <div className="space-y-6">
                <div>
                  <h4 className="text-xl font-bold text-white mb-1">{selectedCase.id}</h4>
                  <p className="text-xs text-[#00f0ff] font-bold uppercase mb-2">State: Field Audit Required</p>
                  <div className="inline-flex items-center space-x-2 bg-slate-900 px-3 py-1.5 border border-slate-800">
                    <Clock className="h-3.5 w-3.5 text-amber-500 animate-pulse" />
                    <span className="text-xs font-mono text-slate-300">Time Active: <span className="text-amber-500 font-bold">{formatTimeActive(selectedCase.timestamp)}</span></span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="bg-slate-900/50 p-3 border border-slate-800">
                    <p className="text-[10px] text-slate-500 uppercase font-bold mb-2">Primary Theft Indicators</p>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-[11px] text-slate-300">Transformer Loss Ratio</span>
                        <span className="text-[11px] text-red-400">+14.2%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[11px] text-slate-300">Phase Unbalance</span>
                        <span className="text-[11px] text-amber-400">High</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[11px] text-slate-300">Consumption Drop (Night)</span>
                        <span className="text-[11px] text-red-400">-42.8%</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-900/50 p-3 border border-slate-800">
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-[10px] text-slate-500 uppercase font-bold">1D Time-Series XAI Heatmap</p>
                      <span className="text-[9px] text-[#00f0ff] uppercase tracking-wider font-bold">Integrated Gradients</span>
                    </div>
                    <div className="bg-black border border-slate-800 rounded overflow-hidden">
                      {/* Live XAI Image generated by the Python backend */}
                      <img src="/xai_heatmap.png" alt="XAI Heatmap" className="w-full object-cover" style={{ filter: 'invert(0.9) hue-rotate(180deg) contrast(1.2)' }} />
                    </div>
                    <p className="text-[9px] text-slate-600 italic mt-2 text-center">Red zones indicate exact temporal sequence driving the theft classification.</p>
                  </div>
                </div>

                <div className="flex flex-col space-y-2 pt-4">
                  <button 
                    onClick={generatePDF}
                    className="w-full py-2 bg-emerald-600/10 border border-emerald-500/30 text-emerald-500 text-xs font-bold uppercase tracking-widest hover:bg-emerald-600/20 transition-all flex items-center justify-center mb-2"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Forensic Audit Report (PDF)
                  </button>
                  <div className="flex space-x-2">
                    <button className="flex-1 py-2 bg-[#1e293b] text-white text-xs font-bold uppercase tracking-widest hover:bg-[#2d3a4f] transition-all">
                      Assign Tech
                    </button>
                    <button 
                      onClick={() => {
                        triggerInspect(selectedCase.id);
                        navigate('/map');
                      }}
                      className="flex-1 py-2 bg-blue-600/10 border border-blue-500/30 text-blue-500 text-xs font-bold uppercase tracking-widest hover:bg-blue-600/20 transition-all flex items-center justify-center"
                    >
                      <MapPin className="h-3 w-3 mr-2" />
                      Inspect Map
                    </button>
                    <button className="flex-1 py-2 bg-red-600/10 border border-red-500/30 text-red-500 text-xs font-bold uppercase tracking-widest hover:bg-red-600/20 transition-all">
                      Flag Fraud
                    </button>
                  </div>
                  <button 
                    onClick={() => {
                      alert("Feedback registered. Tensor sequence sent to ML Engine for Edge XGBoost reinforcement learning.");
                      setSelectedCase(null);
                    }}
                    className="w-full py-2 bg-slate-900 border border-slate-700 text-slate-400 text-xs font-bold uppercase tracking-widest hover:bg-slate-800 hover:text-white transition-all flex justify-center items-center"
                  >
                    Mark as Hardware Failure (False Positive)
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[300px] flex items-center justify-center border border-slate-800 border-dashed rounded-xl bg-slate-950/30">
              <p className="text-[10px] text-slate-600 uppercase font-bold tracking-widest px-12 text-center">
                Select an investigation to view forensic audit data
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
