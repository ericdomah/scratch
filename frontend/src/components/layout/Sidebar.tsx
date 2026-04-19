import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Map, 
  Bell, 
  BarChart2, 
  ShieldAlert, 
  Settings,
  Zap
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const navItems = [
  { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Geospatial View', path: '/map', icon: Map },
  { name: 'Alerts', path: '/alerts', icon: Bell },
  { name: 'Analytics', path: '/analytics', icon: BarChart2 },
  { name: 'Investigations', path: '/investigations', icon: ShieldAlert },
  { name: 'Settings', path: '/settings', icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-[#050505] border-r border-[#1e293b] text-slate-300 flex flex-col h-screen font-mono fixed left-0 top-0">
      <div className="h-16 flex items-center px-6 border-b border-[#1e293b] bg-[#000000]">
        <Zap className="h-6 w-6 text-blue-500 mr-2" />
        <span className="font-bold text-lg text-white tracking-wide">GridGuard AI</span>
      </div>
      
      <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 px-3">
          Control Center
        </div>
        
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) => cn(
                "flex items-center px-3 py-2.5 rounded-none text-sm font-medium transition-colors border-l-2",
                isActive 
                  ? "bg-[#00f0ff]/10 text-[#00f0ff] border-[#00f0ff]" 
                  : "border-transparent hover:bg-[#1e293b] text-slate-400 hover:text-slate-100"
              )}
            >
              <Icon className="mr-3 h-5 w-5" />
              {item.name}
            </NavLink>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-slate-800 text-xs text-slate-500 text-center">
        v2.0.0-rc1 • Secure Connection
      </div>
    </aside>
  );
}
