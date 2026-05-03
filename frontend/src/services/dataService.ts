export const generateMeters = (count: number) => {
  return Array.from({ length: count }, (_, i) => {
    const isBypass = Math.random() > 0.95; // 5% high risk
    const isWarning = !isBypass && Math.random() > 0.85; // 10% medium risk
    
    // Precise TRNC City Clustering to guarantee 100% landmass placement
    const CITIES = [
      { name: 'Lefkoşa', weight: 0.30, lat: 35.1856, lng: 33.3823, varLat: [-0.08, 0.08], varLng: [-0.10, 0.10] },
      { name: 'Girne', weight: 0.15, lat: 35.3325, lng: 33.3166, varLat: [-0.08, 0.01], varLng: [-0.15, 0.15] },
      { name: 'Gazimağusa', weight: 0.15, lat: 35.1149, lng: 33.9392, varLat: [-0.08, 0.08], varLng: [-0.12, 0.02] },
      { name: 'Güzelyurt', weight: 0.10, lat: 35.1997, lng: 32.9915, varLat: [-0.06, 0.06], varLng: [-0.05, 0.10] },
      { name: 'İskele', weight: 0.10, lat: 35.2869, lng: 33.8881, varLat: [-0.10, 0.10], varLng: [-0.10, 0.10] },
      { name: 'Lefke', weight: 0.05, lat: 35.1119, lng: 32.8483, varLat: [-0.03, 0.05], varLng: [0.00, 0.08] },
      { name: 'Karpaz', weight: 0.10, lat: 35.4500, lng: 34.2000, varLat: [-0.10, 0.10], varLng: [-0.30, 0.20] },
      { name: 'Rural Scatters', weight: 0.05, lat: 35.2500, lng: 33.5000, varLat: [-0.20, 0.20], varLng: [-0.40, 0.40] }
    ];

    const r = Math.random();
    let cumulative = 0;
    let selectedCity = CITIES[0];
    
    for (const city of CITIES) {
      cumulative += city.weight;
      if (r <= cumulative) {
        selectedCity = city;
        break;
      }
    }

    // Apply safe variance
    const lat = selectedCity.lat + (Math.random() * (selectedCity.varLat[1] - selectedCity.varLat[0]) + selectedCity.varLat[0]);
    const lng = selectedCity.lng + (Math.random() * (selectedCity.varLng[1] - selectedCity.varLng[0]) + selectedCity.varLng[0]);

    return {
      id: `KIB-TEK-${1000 + i}`,
      lat,
      lng,
      district: selectedCity.name,
      risk: isBypass ? 'high' : isWarning ? 'medium' : 'low',
      confidence: isBypass ? Math.random() * 0.4 + 0.6 : Math.random() * 0.5,
      status: isBypass ? (Math.random() > 0.5 ? 'investigating' : 'pending') : 'cleared',
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString()
    };
  });
};

export const globalMeters = generateMeters(1500);
