export const generateMeters = (count: number) => {
  return Array.from({ length: count }, (_, i) => {
    const isBypass = Math.random() > 0.95; // 5% high risk
    const isWarning = !isBypass && Math.random() > 0.85; // 10% medium risk
    
    // Precise TRNC City Clustering to guarantee 100% landmass placement
    const CITIES = [
      // Lefkoşa (Central, safe all around)
      { weight: 0.35, lat: 35.1856, lng: 33.3823, varLat: [-0.04, 0.04], varLng: [-0.06, 0.06] },
      // Girne (North Coast, must only vary SOUTH and slightly east/west)
      { weight: 0.20, lat: 35.3325, lng: 33.3166, varLat: [-0.04, 0.00], varLng: [-0.08, 0.08] },
      // Gazimağusa (East Coast, must only vary WEST and slightly north/south)
      { weight: 0.15, lat: 35.1149, lng: 33.9392, varLat: [-0.03, 0.03], varLng: [-0.06, 0.00] },
      // Güzelyurt (West Coast, must only vary EAST and slightly north/south)
      { weight: 0.10, lat: 35.1997, lng: 32.9915, varLat: [-0.03, 0.03], varLng: [0.00, 0.05] },
      // İskele (Inland East)
      { weight: 0.10, lat: 35.2869, lng: 33.8881, varLat: [-0.03, 0.03], varLng: [-0.04, 0.04] },
      // Lefke (Deep West)
      { weight: 0.05, lat: 35.1119, lng: 32.8483, varLat: [-0.01, 0.02], varLng: [0.00, 0.03] },
      // Karpaz Peninsula (Extremely thin strip, varying southwest)
      { weight: 0.05, lat: 35.5997, lng: 34.3822, varLat: [-0.04, 0.00], varLng: [-0.15, 0.00] }
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
      risk: isBypass ? 'high' : isWarning ? 'medium' : 'low',
      confidence: isBypass ? Math.random() * 0.4 + 0.6 : Math.random() * 0.5,
      status: isBypass ? (Math.random() > 0.5 ? 'investigating' : 'pending') : 'cleared',
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString()
    };
  });
};

export const globalMeters = generateMeters(1500);
