import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import AuthGuard from './components/auth/AuthGuard';
import Login from './pages/Login';

import GeospatialMap from './pages/GeospatialMap';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';

// Placeholder Pages
const Alerts = () => <div className="p-6"><h2 className="text-2xl font-bold">Alerts</h2><p className="text-slate-400 mt-2">Tabular alert view goes here.</p></div>;

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        
        <Route element={<AuthGuard />}>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="map" element={<GeospatialMap />} />
            <Route path="alerts" element={<Alerts />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="investigations" element={<div className="p-6 text-xl">Investigations</div>} />
            <Route path="settings" element={<div className="p-6 text-xl">Settings</div>} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
