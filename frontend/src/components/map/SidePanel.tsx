import React, { useState } from 'react';
import { X, Activity, ServerCrash, AlertTriangle, ShieldCheck, Power, PowerOff, Download } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { useAuthStore } from '../../store/authStore';

interface SidePanelProps {
  meter: any;
  onClose: () => void;
}

export default function SidePanel({ meter, onClose }: SidePanelProps) {
  const { user } = useAuthStore();
  const [isPoweredOn, setIsPoweredOn] = useState(true);
  const isAdmin = user?.role === 'Admin';
  
  if (!meter) return null;

  // Mock data for the charts
  const timeSeriesData = Array.from({ length: 24 }, (_, i) => ({
    time: `${i}:00`,
    usage: meter.risk === 'high' && i >= 2 && i <= 5 ? Math.random() * 0.5 : Math.random() * 5 + 2
  }));

  const shapData = [
    { feature: 'Night_Usage', importance: 0.8 },
    { feature: 'Voltage_Drop', importance: 0.6 },
    { feature: 'Temp_Variance', importance: 0.3 },
    { feature: 'Weekend_Avg', importance: 0.1 },
  ];

  const exportMeterReport = () => {
    const doc = new jsPDF();
    
    // Header
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 0, 210, 40, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.text("METER INVESTIGATION MEMO", 14, 25);
    
    doc.setFontSize(10);
    doc.text(`Institutional Document | KIB-TEK Authority TRNC`, 14, 32);

    // Body Info
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.text(`Meter ID: ${meter.id}`, 14, 55);
    
    const summaryData = [
      ['Metric', 'Assigned Value'],
      ['Geospatial Coordinates', `${meter.lat.toFixed(6)}, ${meter.lng.toFixed(6)}`],
      ['Theft Risk Level', meter.risk.toUpperCase()],
      ['AI Detection Confidence', `${(meter.confidence * 100).toFixed(1)}%`],
      ['Operational Status', meter.status.toUpperCase()],
      ['Remote Power State', isPoweredOn ? 'ACTIVE (Connected)' : 'DISCONNECTED (Remote-Off)'],
    ];

    autoTable(doc, {
      startY: 65,
      head: [summaryData[0]],
      body: summaryData.slice(1),
      theme: 'grid',
      headStyles: { fillColor: [15, 23, 42] },
    });

    // Auditor Signature Block
    const auditY = (doc as any).lastAutoTable.finalY + 30;
    doc.setFontSize(12);
    doc.text("Auditor Diagnostic Note:", 14, auditY);
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(meter.risk === 'high' 
      ? "STATISTICAL ANOMALY DETECTED. IMMEDIATE PHYSICAL INSPECTION REQUIRED."
      : "No significant deviation from utility benchmarks.", 14, auditY + 10);
    
    doc.setDrawColor(200);
    doc.line(14, auditY + 40, 70, auditY + 40);
    doc.text("E-Signature: KIB-TEK AI ENGINE", 14, auditY + 45);

    // High-Priority Data-URI Stream
    const dataUri = doc.output('datauristring');
    const link = document.createElement('a');
    link.href = dataUri;
    link.download = `KIB_TEK_Report_${meter.id}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col h-full shadow-2xl overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex justify-between items-start sticky top-0 bg-slate-900/95 backdrop-blur-sm z-10">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center">
            {meter.id}
          </h2>
          <p className="text-sm text-slate-400 mt-1">Lat: {meter.lat.toFixed(4)} | Lng: {meter.lng.toFixed(4)}</p>
        </div>
        <button onClick={onClose} className="p-1 text-slate-400 hover:text-white bg-slate-800 rounded-md">
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Snapshot Cards */}
      <div className="p-4 grid grid-cols-2 gap-4 border-b border-slate-800">
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
          <span className="text-xs text-slate-400 uppercase font-semibold">Theft Probability</span>
          <div className="mt-1 flex items-center">
            <span className={`text-2xl font-bold ${meter.risk === 'high' ? 'text-red-500' : meter.risk === 'medium' ? 'text-amber-500' : 'text-green-500'}`}>
              {(meter.confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
          <span className="text-xs text-slate-400 uppercase font-semibold">Power State</span>
          <div className="mt-1 flex items-center">
            {isPoweredOn ? (
              <>
                <Power className="h-5 w-5 text-emerald-500 mr-2" />
                <span className="text-sm font-bold text-emerald-500 uppercase">Active</span>
              </>
            ) : (
              <>
                <PowerOff className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-sm font-bold text-red-500 uppercase">Off-Grid</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Time Series Preview */}
      <div className="p-4 border-b border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center">
          <Activity className="h-4 w-4 mr-2" /> 24-Hour Consumption Trend
        </h3>
        <div className="h-40 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickMargin={5} />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Line type="monotone" dataKey="usage" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* SHAP / XAI Chart */}
      <div className="p-4 border-b border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center">
          <ServerCrash className="h-4 w-4 mr-2" /> SHAP Feature Importance
        </h3>
        <div className="h-40 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis type="number" stroke="#64748b" fontSize={10} />
              <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={10} width={80} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }}
              />
              <Bar dataKey="importance" fill="#ef4444" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* AI Explanation NLP */}
      <div className="p-4 flex-1">
        <h3 className="text-sm font-semibold text-slate-300 mb-2">AI Diagnostic Summary</h3>
        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 border-l-2 border-l-red-500">
          <p className="text-sm text-slate-400">
            {meter.risk === 'high' 
              ? "Critical anomaly detected. Significant drop in consumption between 02:00–05:00 strongly correlates with historical bypass markers. SHAP analysis confirms Night_Usage drop as primary statistical driver."
              : "Meter operating within normal seasonal parameters. Minor voltage fluctuations align with regional grid variance."}
          </p>
        </div>

        <div className="mt-6 flex space-x-3">
          <button 
            onClick={exportMeterReport}
            className="flex-1 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md text-sm font-medium border border-slate-700 transition-colors flex items-center justify-center"
          >
            <Download className="h-4 w-4 mr-2" />
            Report
          </button>
          
          {isAdmin ? (
            <button 
              onClick={() => setIsPoweredOn(!isPoweredOn)}
              className={`flex-1 py-1.5 rounded-md text-sm font-bold border transition-all flex items-center justify-center ${
                isPoweredOn 
                  ? 'bg-red-600/10 hover:bg-red-600/30 text-red-500 border-red-500/30' 
                  : 'bg-emerald-600/10 hover:bg-emerald-600/30 text-emerald-500 border-emerald-500/30'
              }`}
            >
              {isPoweredOn ? <PowerOff className="h-4 w-4 mr-2" /> : <Power className="h-4 w-4 mr-2" />}
              {isPoweredOn ? 'Shutoff' : 'Restore'}
            </button>
          ) : (
            <button 
              disabled 
              title="Admin privileges required"
              className="flex-1 py-1.5 bg-slate-900/50 text-slate-600 rounded-md text-sm font-medium border border-slate-800 cursor-not-allowed flex items-center justify-center"
            >
              <PowerOff className="h-4 w-4 mr-2" />
              Locked
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
