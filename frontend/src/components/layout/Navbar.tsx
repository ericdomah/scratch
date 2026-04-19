import React from 'react';
import { Bell, LogOut, User } from 'lucide-react';
import { useAuthStore } from '../../store/authStore';

export default function Navbar() {
  const { user, logout } = useAuthStore();

  return (
    <header className="h-16 bg-[#000000] border-b border-[#1e293b] flex items-center justify-between px-6 sticky top-0 z-20 font-mono">
      <div className="flex items-center">
        <h1 className="text-xl font-semibold text-white tracking-tight">System Overview</h1>
      </div>
      
      <div className="flex items-center space-x-4">
        <button className="p-2 text-slate-400 hover:text-[#00f0ff] rounded-none hover:bg-[#1e293b] transition-colors relative">
          <Bell className="h-5 w-5" />
          <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-none"></span>
        </button>
        
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
