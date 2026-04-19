import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Zap, Lock, ShieldCheck, Activity, Globe } from 'lucide-react';
import { useAuthStore } from '../store/authStore';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [bootSequence, setBootSequence] = useState(0);
  const login = useAuthStore(state => state.login);
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setInterval(() => {
      setBootSequence(prev => (prev < 100 ? prev + 2 : 100));
    }, 30);
    return () => clearInterval(timer);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    setTimeout(() => {
      if (email === 'admin@gridguard.ai' && password === 'admin') {
        login('mock_jwt_token_admin_123', {
          id: 'u1',
          email: 'admin@gridguard.ai',
          name: 'Operator 01',
          role: 'Admin'
        });
        navigate('/dashboard');
      } else if (email === 'analyst@gridguard.ai' && password === 'analyst') {
        login('mock_jwt_token_analyst_123', {
          id: 'u2',
          email: 'analyst@gridguard.ai',
          name: 'Analyst 02',
          role: 'Analyst'
        });
        navigate('/dashboard');
      } else {
        setError('AUTH_FAILURE: ACCESS_DENIED. Check credentials.');
      }
      setIsLoading(false);
    }, 1200);
  };

  return (
    <div className="min-h-screen bg-[#020202] flex items-center justify-center font-mono overflow-hidden relative">
      {/* Tactical Background Elements */}
      <div className="absolute inset-0 opacity-10 pointer-events-none">
        <div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(#1e293b 1px, transparent 1px)', backgroundSize: '32px 32px' }}></div>
        <div className="absolute top-0 left-0 w-full h-1 bg-[#00f0ff] animate-[scan_4s_linear_infinite]"></div>
      </div>

      {/* Main Login Card */}
      <div className="w-full max-w-[400px] z-10 p-1">
        <div className="bg-[#050505] border border-[#1e293b] p-8 relative overflow-hidden">
          {/* Edge Accents */}
          <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-[#00f0ff]"></div>
          <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-[#00f0ff]"></div>
          
          {/* Header */}
          <div className="mb-10">
            <div className="flex items-center space-x-3 mb-2">
              <Zap className="h-6 w-6 text-[#00f0ff]" />
              <span className="text-xl font-black text-white tracking-tighter uppercase">GridGuard AI</span>
            </div>
            <div className="h-px w-full bg-[#1e293b] mb-4"></div>
            <div className="flex justify-between items-center text-[10px] uppercase font-bold text-slate-500">
              <span className="flex items-center"><Globe className="h-3 w-3 mr-1 text-emerald-500" /> KIB-TEK Authorized Access</span>
              <span>TRNC Region: 05</span>
            </div>
          </div>

          <form className="space-y-6" onSubmit={handleLogin}>
            {error && (
              <div className="bg-red-950/30 border border-red-500/50 text-red-500 text-[11px] p-3 rounded-none flex items-start animate-pulse">
                <ShieldAlert className="h-4 w-4 mr-2 shrink-0" />
                <span>{error}</span>
              </div>
            )}
            
            <div className="space-y-4">
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
                  Security Identifier
                </label>
                <input
                  type="email"
                  required
                  className="w-full bg-[#0a0a0a] border border-[#1e293b] px-4 py-3 text-sm text-white focus:outline-none focus:border-[#00f0ff] transition-all rounded-none placeholder-slate-700"
                  placeholder="ID_SEQUENCE@GRIDGUARD.AI"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>

              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
                  Access Passcode
                </label>
                <div className="relative">
                  <input
                    type="password"
                    required
                    className="w-full bg-[#0a0a0a] border border-[#1e293b] px-4 py-3 text-sm text-white focus:outline-none focus:border-[#00f0ff] transition-all rounded-none placeholder-slate-700"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <Lock className="absolute right-4 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-700" />
                </div>
              </div>
            </div>

            <div className="pt-4">
              <button
                type="submit"
                disabled={isLoading}
                className="group relative w-full bg-[#00f0ff] hover:bg-[#00f0ff]/90 text-black font-black py-4 transition-all disabled:opacity-50 overflow-hidden"
              >
                <span className="relative z-10 flex items-center justify-center uppercase text-sm tracking-widest">
                  {isLoading ? 'Decrypting...' : 'Initiate Secure Uplink'}
                </span>
                <div className="absolute inset-0 bg-white/20 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-500"></div>
              </button>
            </div>
          </form>

          {/* Footer Info */}
          <div className="mt-10 border-t border-[#1e293b] pt-6 space-y-4">
            <div className="flex justify-between text-[10px] text-slate-500 uppercase font-bold">
              <span>Boot Sequence</span>
              <span className="text-[#00f0ff]">{bootSequence}%</span>
            </div>
            <div className="w-full bg-[#0a0a0a] h-1">
              <div className="bg-[#00f0ff] h-full transition-all duration-300" style={{ width: `${bootSequence}%` }}></div>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex space-x-2">
                <Activity className="h-3 w-3 text-emerald-500 animate-pulse" />
                <span className="text-[10px] text-slate-600 uppercase font-bold">Grid_Uplink_Online</span>
              </div>
              <span className="text-[9px] text-slate-700">AUTH_v4.2.0</span>
            </div>
          </div>
        </div>
        
        {/* Test Credits - Subdued */}
        <p className="mt-4 text-[9px] text-slate-700 uppercase tracking-widest text-center leading-relaxed">
          Operational Use Only. Authorized by the Northern Cyprus Utility Protection Agency (NCUPA).
          <br/>
          Test: admin@gridguard.ai | admin
        </p>
      </div>
      
      {/* CSS Animations */}
      <style>{`
        @keyframes scan {
          0% { top: -5%; }
          100% { top: 105%; }
        }
      `}</style>
    </div>
  );
}
