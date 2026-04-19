import React, { useState, useRef, useEffect } from 'react';
import { Bell, LogOut, User, ShieldAlert, Clock } from 'lucide-react';
import { useAuthStore } from '../../store/authStore';
import { useGridStore } from '../../store/gridStore';
import { useNavigate } from 'react-router-dom';

export default function Navbar() {
  const { user, logout } = useAuthStore();
  const { liveAlerts, setSelectedMeterId } = useGridStore();
  const [showNotifications, setShowNotifications] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowNotifications(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleNotificationClick = (id: string) => {
    setSelectedMeterId(id);
    setShowNotifications(false);
    navigate('/map');
  };

  return (
    <header className="h-16 bg-[#000000] border-b border-[#1e293b] flex items-center justify-between px-6 sticky top-0 z-50 font-mono">
      <div className="flex items-center">
        <h1 className="text-xl font-semibold text-white tracking-tight">System Overview</h1>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="relative" ref={dropdownRef}>
          <button 
            onClick={() => setShowNotifications(!showNotifications)}
            className={`p-2 rounded-none transition-colors relative ${showNotifications ? 'bg-[#1e293b] text-[#00f0ff]' : 'text-slate-400 hover:text-[#00f0ff] hover:bg-[#1e293b]'}`}
          >
            <Bell className="h-5 w-5" />
            {liveAlerts.length > 0 && (
              <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-none animate-pulse"></span>
            )}
          </button>

          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-[#050505] border border-slate-800 shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2">
              <div className="px-4 py-3 border-b border-slate-800 bg-[#0a0a0a] flex justify-between items-center">
                <span className="text-[10px] font-bold text-white uppercase tracking-widest">Grid Anomaly Feed</span>
                <span className="text-[9px] text-[#00f0ff] font-bold">{liveAlerts.length} Active</span>
              </div>
              <div className="max-h-[400px] overflow-y-auto">
                {liveAlerts.slice(0, 10).map((alert, i) => (
                  <div 
                    key={i} 
                    onClick={() => handleNotificationClick(alert.id)}
                    className="px-4 py-3 border-b border-slate-800/50 hover:bg-[#1e293b]/30 cursor-pointer transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[11px] font-bold text-white font-mono">{alert.id}</span>
                      <span className={`text-[9px] font-black uppercase ${alert.risk === 'high' ? 'text-red-500' : 'text-amber-500'}`}>
                        {alert.risk}
                      </span>
                    </div>
                    <div className="flex items-center text-[10px] text-slate-500">
                      <Clock className="h-3 w-3 mr-1" />
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
                {liveAlerts.length === 0 && (
                  <div className="p-8 text-center text-[10px] text-slate-600 uppercase font-bold tracking-widest">
                    No anomalies detected
                  </div>
                )}
              </div>
              <div 
                onClick={() => { navigate('/alerts'); setShowNotifications(false); }}
                className="px-4 py-2 text-center text-[10px] font-bold text-slate-500 hover:text-white bg-[#0a0a0a] border-t border-slate-800 cursor-pointer uppercase transition-colors"
              >
                View All Alerts
              </div>
            </div>
          )}
        </div>
        
        <div className="h-6 w-px bg-slate-700 mx-2"></div>
        
        <div className="flex items-center space-x-3">
          <div className="flex flex-col items-end">
            <span className="text-sm font-medium text-white">{user?.name || "Unknown Operator"}</span>
            <span className="text-xs text-blue-400 font-medium">{user?.role || "Restricted"}</span>
          </div>
          <div className="h-9 w-9 rounded-none bg-[#050505] flex items-center justify-center border border-[#1e293b] overflow-hidden">
            {user?.role === 'Admin' ? (
              <div className="bg-[#00f0ff]/20 text-[#00f0ff] font-bold w-full h-full flex items-center justify-center text-sm">A</div>
            ) : (
              <User className="h-5 w-5 text-slate-400" />
            )}
          </div>
        </div>
        
        <button 
          onClick={logout}
          className="p-2 ml-2 text-slate-400 hover:text-red-400 rounded-none hover:bg-[#1e293b] transition-colors" 
          title="Logout"
        >
          <LogOut className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}
