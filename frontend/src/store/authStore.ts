import { create } from 'zustand';

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'Admin' | 'Analyst';
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
}

// Retrieve initial state from secure local storage
const token = localStorage.getItem('gridguard_token');
const userStr = localStorage.getItem('gridguard_user');
const user = userStr ? JSON.parse(userStr) : null;

export const useAuthStore = create<AuthState>((set) => ({
  user: user,
  token: token,
  isAuthenticated: !!token,
  
  login: (token, user) => {
    localStorage.setItem('gridguard_token', token);
    localStorage.setItem('gridguard_user', JSON.stringify(user));
    set({ token, user, isAuthenticated: true });
  },
  
  logout: () => {
    localStorage.removeItem('gridguard_token');
    localStorage.removeItem('gridguard_user');
    set({ token: null, user: null, isAuthenticated: false });
  }
}));
