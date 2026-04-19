import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Shield, Zap, Database, Save, CheckCircle, RefreshCw } from 'lucide-react';
import { useGridStore } from '../store/gridStore';

export default function Settings() {
  const { config, updateConfig } = useGridStore();
  const [localConfig, setLocalConfig] = useState(config);
  const [isDeploying, setIsDeploying] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<'none' | 'success' | 'error'>('none');

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleInputChange = (key: keyof typeof config, value: string | number) => {
    setLocalConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleToggle = (key: keyof typeof config) => {
    setLocalConfig(prev => ({ ...prev, [key]: !prev[key as keyof typeof config] }));
  };

  const handleTestConnection = () => {
    setIsTesting(true);
    setTestResult('none');
    setTimeout(() => {
      setIsTesting(false);
      if (localConfig.relayProtocol === 'Simulated Data Stream') {
        setTestResult('success');
      } else {
        setTestResult('error');
      }
      setTimeout(() => setTestResult('none'), 4000);
    }, 1500);
  };

  const handleDeploy = () => {
    setIsDeploying(true);
    setTimeout(() => {
      updateConfig(localConfig);
      setIsDeploying(false);
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    }, 1200);
  };

  return (
    <div className="space-y-8 max-w-4xl pb-20 relative">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center font-mono">
          <SettingsIcon className="h-6 w-6 mr-3 text-slate-500" />
          Grid Control Settings
        </h2>
        <p className="text-sm text-slate-400 mt-1">Configure systemic detection logic and secure infrastructure parameters.</p>
      </div>

      <div className="grid gap-6">
        {/* ML Engine Section */}
        <div className="bg-[#050505] border border-slate-800 rounded-none overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-800 bg-[#0a0a0a] flex items-center">
            <Shield className="h-4 w-4 mr-3 text-[#00f0ff]" />
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Hybrid ML Detection Engine</h3>
              <p className="text-[10px] text-slate-500 font-medium">Adjust sensitivity and ensemble weights for the Super-Hybrid architecture.</p>
            </div>
          </div>
          <div className="p-6 space-y-6">
            {[
              { label: 'Ensemble Threshold', key: 'ensembleThreshold', max: 1 },
              { label: 'XGBoost Weight', key: 'xgboostWeight', max: 1 },
              { label: 'Transformer Weight', key: 'transformerWeight', max: 1 },
              { label: 'Bi-LSTM Weight', key: 'bilstmWeight', max: 1 },
              { label: 'TFT Weight', key: 'tftWeight', max: 1 },
            ].map(({ label, key, max }) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{label}</span>
                <div className="flex items-center space-x-3">
                  <input 
                    type="range" 
                    min="0" 
                    max={max} 
                    step="0.01"
                    value={localConfig[key as keyof typeof localConfig] as number}
                    onChange={(e) => handleInputChange(key as keyof typeof config, parseFloat(e.target.value))}
                    className="w-32 accent-[#00f0ff]" 
                  />
                  <span className="text-[11px] text-white font-mono w-8">
                    {(localConfig[key as keyof typeof localConfig] as number).toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
            
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Imbalance Penalty (Pos)</span>
              <input 
                type="number" 
                value={localConfig.imbalancePenalty}
                onChange={(e) => handleInputChange('imbalancePenalty', parseFloat(e.target.value))}
                className="bg-black border border-slate-800 text-xs text-white px-3 py-1.5 focus:outline-none focus:border-[#00f0ff] font-mono min-w-[80px] text-right"
              />
            </div>
          </div>
        </div>

        {/* Connectivity Section */}
        <div className="bg-[#050505] border border-slate-800 rounded-none overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-800 bg-[#0a0a0a] flex items-center justify-between">
            <div className="flex items-center">
              <Zap className="h-4 w-4 mr-3 text-[#00f0ff]" />
              <div>
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">System Connectivity</h3>
                <p className="text-[10px] text-slate-500 font-medium">Link the AI engine to active field infrastructure or telemetry streams.</p>
              </div>
            </div>
            <button 
              onClick={handleTestConnection}
              disabled={isTesting}
              className="flex items-center px-4 py-1.5 bg-[#1e293b] text-white font-bold uppercase text-[10px] tracking-widest hover:bg-[#2d3a4f] transition-all disabled:opacity-50 border border-slate-700"
            >
              <RefreshCw className={`h-3 w-3 mr-2 ${isTesting ? 'animate-spin' : ''}`} />
              Test Uplink
            </button>
          </div>
          
          {testResult === 'error' && (
            <div className="bg-red-950/30 border-b border-red-500/50 text-red-500 text-[11px] p-3 flex items-start animate-pulse font-mono">
              [CONNECTION_REFUSED] Hardware Relay not found at provided endpoint. Fallback to Simulation recommended.
            </div>
          )}
          {testResult === 'success' && (
            <div className="bg-emerald-950/30 border-b border-emerald-500/50 text-emerald-500 text-[11px] p-3 flex items-start font-mono">
              [UPLINK_ESTABLISHED] Receiving synthetic telemetry stream...
            </div>
          )}

          <div className="p-6 space-y-6">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">WebSocket Server URL</span>
              <input 
                type="text" 
                value={localConfig.wsUrl} 
                onChange={(e) => handleInputChange('wsUrl', e.target.value)}
                className="bg-black border border-slate-800 text-xs text-white px-3 py-1.5 focus:outline-none focus:border-[#00f0ff] font-mono w-[300px]" 
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Secure Relay Key</span>
              <input 
                type="password" 
                value={localConfig.apiKey} 
                onChange={(e) => handleInputChange('apiKey', e.target.value)}
                className="bg-black border border-slate-800 text-xs text-white px-3 py-1.5 focus:outline-none focus:border-[#00f0ff] font-mono w-[300px]" 
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Relay Protocol</span>
              <select 
                value={localConfig.relayProtocol}
                onChange={(e) => handleInputChange('relayProtocol', e.target.value)}
                className="bg-black border border-slate-800 text-xs text-[#00f0ff] px-3 py-1.5 focus:outline-none focus:border-[#00f0ff] font-mono min-w-[200px]"
              >
                <option value="Simulated Data Stream">Simulated Data Stream</option>
                <option value="PLC / AES-256">PLC / AES-256 (KIB-TEK Standard)</option>
                <option value="IEC 61850">IEC 61850 (Substation)</option>
                <option value="DNP3">DNP3 (Legacy)</option>
                <option value="MQTT-SN">MQTT-SN (IoT Devices)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Reporting Section */}
        <div className="bg-[#050505] border border-slate-800 rounded-none overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-800 bg-[#0a0a0a] flex items-center">
            <Database className="h-4 w-4 mr-3 text-[#00f0ff]" />
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Reporting & Logs</h3>
              <p className="text-[10px] text-slate-500 font-medium">Manage persistence layers and automated export formats.</p>
            </div>
          </div>
          <div className="p-6 space-y-6">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Auto-Export PDF Reports</span>
              <div 
                onClick={() => handleToggle('autoExport')}
                className={`w-10 h-5 border rounded-none relative cursor-pointer transition-colors ${localConfig.autoExport ? 'bg-[#00f0ff]/20 border-[#00f0ff]/50' : 'bg-slate-900 border-slate-700'}`}
              >
                <div className={`absolute top-1 w-3 h-3 transition-all ${localConfig.autoExport ? 'right-1 bg-[#00f0ff]' : 'left-1 bg-slate-500'}`}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end pt-4 sticky bottom-6 z-10">
        <button 
          onClick={handleDeploy}
          disabled={isDeploying || isTesting}
          className="flex items-center px-8 py-3 bg-[#00f0ff] text-black font-black uppercase text-xs tracking-widest hover:bg-[#00f0ff]/80 transition-all disabled:opacity-50"
        >
          {isDeploying ? (
            <span className="animate-pulse">Deploying to Edge...</span>
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              Deploy Configuration
            </>
          )}
        </button>
      </div>

      {showSuccess && (
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-emerald-500/10 border border-emerald-500 text-emerald-500 px-6 py-3 font-mono text-xs uppercase tracking-widest flex items-center animate-in slide-in-from-bottom-5 z-50 shadow-[0_0_15px_rgba(16,185,129,0.3)]">
          <CheckCircle className="h-4 w-4 mr-2" />
          Configuration deployed to active network
        </div>
      )}
    </div>
  );
}
