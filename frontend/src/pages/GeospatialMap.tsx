import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import SidePanel from '../components/map/SidePanel';
import { globalMeters } from '../services/dataService';
import { useGridStore } from '../store/gridStore';

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

function MapController({ inspectEvent, setSelectedMeter }: { inspectEvent: { meterId: string, timestamp: number } | null, setSelectedMeter: (m: any) => void }) {
  const map = useMap();

  useEffect(() => {
    if (inspectEvent) {
      const meter = globalMeters.find(m => m.id === inspectEvent.meterId);
      if (meter) {
        // Use a slight timeout to ensure map is ready after navigation
        const timer = setTimeout(() => {
          map.flyTo([meter.lat, meter.lng], 16, { 
            duration: 1.5,
            animate: true
          });
          setSelectedMeter(meter);
        }, 150);
        return () => clearTimeout(timer);
      }
    }
  }, [inspectEvent, map, setSelectedMeter]);

  return null;
}

export default function GeospatialMap() {
  const [selectedMeter, setSelectedMeter] = useState<any>(null);
  const { inspectEvent, liveAlerts } = useGridStore();

  return (
    <div className="flex h-full min-h-[500px] w-full border border-slate-800 rounded-xl overflow-hidden relative shadow-2xl">
      <div className="flex-1 relative bg-slate-900 z-0">
        <MapContainer 
          center={[35.25, 33.3]} 
          zoom={9} 
          className="w-full h-full z-0"
          zoomControl={false}
        >
          <MapController inspectEvent={inspectEvent} setSelectedMeter={setSelectedMeter} />
          
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <MarkerClusterGroup
            chunkedLoading
            maxClusterRadius={50}
          >
            {globalMeters.map((meter) => (
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

          {/* High-Visibility Focus Ring for Inspected Meter */}
          {selectedMeter && (
            <Marker 
              position={[selectedMeter.lat, selectedMeter.lng]}
              icon={L.divIcon({
                className: 'focus-marker',
                html: `<div class="w-8 h-8 -ml-2.5 -mt-2.5 rounded-full border-2 border-[#00f0ff] animate-ping opacity-75"></div>`,
                iconSize: [0, 0]
              })}
              zIndexOffset={1000}
            />
          )}
        </MapContainer>
      </div>

      {/* Contextual Side Panel containing XAI Visualizations */}
      <div className={`transition-all duration-300 ease-in-out bg-slate-900 border-l border-slate-800 ${selectedMeter ? 'max-w-md w-96 opacity-100' : 'max-w-0 w-0 opacity-0 overflow-hidden'}`}>
        <SidePanel 
          meter={selectedMeter ? {
            ...selectedMeter,
            ...liveAlerts.find(a => a.id === selectedMeter.id)
          } : null} 
          onClose={() => setSelectedMeter(null)} 
        />
      </div>
    </div>
  );
}
