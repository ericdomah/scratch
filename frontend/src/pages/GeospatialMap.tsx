import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import SidePanel from '../components/map/SidePanel';

// Mock smart meter generation representing SGCC dataset context
const mockMeters = Array.from({ length: 450 }, (_, i) => {
  const isBypass = Math.random() > 0.95; // 5% high risk
  const isWarning = !isBypass && Math.random() > 0.85; // 10% medium risk
  return {
    id: `SGCC-${Math.floor(1000 + Math.random() * 9000)}-${i}`,
    lat: 39.9042 + (Math.random() - 0.5) * 0.3, // Beijing area coordinates
    lng: 116.4074 + (Math.random() - 0.5) * 0.3,
    risk: isBypass ? 'high' : isWarning ? 'medium' : 'low',
    confidence: isBypass ? Math.random() * 0.4 + 0.6 : Math.random() * 0.5,
    status: isBypass ? (Math.random() > 0.5 ? 'investigating' : 'pending') : 'cleared'
  };
});

// Custom DOM icon factory for Risk statuses
const createRiskIcon = (risk: string) => {
  const colorClass = 
    risk === 'high' ? 'bg-red-500 shadow-red-500/80 animate-pulse' : 
    risk === 'medium' ? 'bg-amber-500 shadow-amber-500/80' : 
    'bg-green-500 shadow-green-500/20 opacity-30';
    
  return L.divIcon({
    className: 'custom-leaflet-marker',
    html: `<div class="w-3 h-3 rounded-full ${colorClass} border border-white shadow-lg"></div>`,
    iconSize: [12, 12],
    iconAnchor: [6, 6]
  });
};

export default function GeospatialMap() {
  const [selectedMeter, setSelectedMeter] = useState<any>(null);

  return (
    <div className="flex h-[calc(100vh-130px)] border border-slate-800 rounded-xl overflow-hidden relative shadow-2xl">
      <div className="flex-1 relative bg-slate-900 z-0">
        <MapContainer 
          center={[39.9042, 116.4074]} 
          zoom={11} 
          className="w-full h-full z-0"
          zoomControl={false}
        >
          {/* Deep dark enterprise map style via Carto CartoDB */}
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <MarkerClusterGroup
            chunkedLoading
            maxClusterRadius={50}
          >
            {mockMeters.map((meter) => (
              <Marker 
                key={meter.id} 
                position={[meter.lat, meter.lng]}
                icon={createRiskIcon(meter.risk)}
                eventHandlers={{
                  click: () => setSelectedMeter(meter)
                }}
              />
            ))}
          </MarkerClusterGroup>
        </MapContainer>
      </div>

      {/* Contextual Side Panel containing XAI Visualizations */}
      <div className={`transition-all duration-300 ease-in-out bg-slate-900 border-l border-slate-800 ${selectedMeter ? 'max-w-md w-96 opacity-100' : 'max-w-0 w-0 opacity-0 overflow-hidden'}`}>
        <SidePanel meter={selectedMeter} onClose={() => setSelectedMeter(null)} />
      </div>
    </div>
  );
}
