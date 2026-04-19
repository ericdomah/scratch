import { create } from 'zustand';
import { globalMeters } from '../services/dataService';

interface GridState {
  totalMeters: number;
  theftAlerts: number;
  investigations: number;
  estimatedLoss: number;
  liveAlerts: any[];
  inspectEvent: { meterId: string, timestamp: number } | null;
  config: {
    ensembleThreshold: number;
    xgboostWeight: number;
    transformerWeight: number;
    bilstmWeight: number;
    tftWeight: number;
    imbalancePenalty: number;
    wsUrl: string;
    relayProtocol: string;
    apiKey: string;
    autoExport: boolean;
  };
  
  // Actions
  addLiveAlert: (alert: any) => void;
  incrementInvestigations: () => void;
  updateKPIs: (data: Partial<GridState>) => void;
  triggerInspect: (id: string) => void;
  updateConfig: (newConfig: Partial<GridState['config']>) => void;
}

const initialAlerts = globalMeters.filter(m => m.risk !== 'low').slice(0, 20);
const initialLoss = initialAlerts.reduce((total, a) => {
  if (a.risk === 'high') return total + 15000;
  if (a.risk === 'medium') return total + 5000;
  return total;
}, 0);

export const useGridStore = create<GridState>((set) => ({
  totalMeters: 1500,
  theftAlerts: 77,
  investigations: 26,
  estimatedLoss: initialLoss, 
  liveAlerts: initialAlerts, 
  inspectEvent: null,
  config: {
    ensembleThreshold: 0.85,
    xgboostWeight: 0.40,
    transformerWeight: 0.30,
    bilstmWeight: 0.20,
    tftWeight: 0.10,
    imbalancePenalty: 11.0,
    wsUrl: 'ws://localhost:8000/ws/telemetry',
    relayProtocol: 'Simulated Data Stream',
    apiKey: '************************',
    autoExport: true,
  },

  addLiveAlert: (alert) => set((state) => {
    const newAlerts = [alert, ...state.liveAlerts].slice(0, 50);
    // Dynamic financial loss calculation
    const newLoss = newAlerts.reduce((total, a) => {
      if (a.risk === 'high') return total + 15000;
      if (a.risk === 'medium') return total + 5000;
      return total;
    }, 0);

    return {
      liveAlerts: newAlerts,
      theftAlerts: state.theftAlerts + (alert.risk === 'high' ? 1 : 0),
      estimatedLoss: newLoss
    };
  }),

  incrementInvestigations: () => set((state) => ({
    investigations: state.investigations + 1
  })),

  updateKPIs: (data) => set((state) => ({
    ...state,
    ...data
  })),

  triggerInspect: (id) => set({ inspectEvent: { meterId: id, timestamp: Date.now() } }),

  updateConfig: (newConfig) => set((state) => ({
    config: { ...state.config, ...newConfig }
  }))
}));
