import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Zap, Lock, Mail, ShieldAlert } from 'lucide-react';
import { useAuthStore } from '../store/authStore';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const login = useAuthStore(state => state.login);
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      // Mocking the FastAPI token response for now
      // In production: await api.post('/auth/token', form_data)
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
          setError('Invalid credentials. Try admin@gridguard.ai / admin');
        }
        setIsLoading(false);
      }, 800);
    } catch (err) {
      setError('Authentication failed. Please check your network.');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center">
          <div className="h-16 w-16 bg-blue-600/20 rounded-full flex items-center justify-center border border-blue-500/30 shadow-[0_0_15px_rgba(59,130,246,0.5)]">
            <Zap className="h-8 w-8 text-blue-400" />
          </div>
        </div>
        <h2 className="mt-6 text-center text-3xl font-extrabold text-white tracking-tight">
          GridGuard AI
        </h2>
        <p className="mt-2 text-center text-sm text-slate-400">
          Secure Central Command Authorization
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-slate-900 py-8 px-4 shadow-2xl shadow-blue-900/10 sm:rounded-xl sm:px-10 border border-slate-800 backdrop-blur-xl">
          <form className="space-y-6" onSubmit={handleLogin}>
            {error && (
              <div className="bg-red-500/10 border border-red-500/50 text-red-500 text-sm p-3 rounded-md flex items-center">
                <ShieldAlert className="h-4 w-4 mr-2" />
                {error}
              </div>
            )}
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Operator ID (Email)
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-slate-500" />
                </div>
                <input
                  type="email"
                  required
                  className="appearance-none block w-full pl-10 px-3 py-2 border border-slate-700 rounded-md shadow-sm placeholder-slate-500 bg-slate-800/50 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm transition-all"
                  placeholder="admin@gridguard.ai"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Security Passcode
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-slate-500" />
                </div>
                <input
                  type="password"
                  required
                  className="appearance-none block w-full pl-10 px-3 py-2 border border-slate-700 rounded-md shadow-sm placeholder-slate-500 bg-slate-800/50 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm transition-all"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 focus:ring-offset-slate-900 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Authenticating...' : 'Establish Secure Connection'}
              </button>
            </div>
            
            <div className="text-center mt-4">
              <span className="text-xs text-slate-500">Test Accounts: admin@gridguard.ai (admin) | analyst@gridguard.ai (analyst)</span>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
