export const generateMeters = (count: number) => {
  return Array.from({ length: count }, (_, i) => {
    const isBypass = Math.random() > 0.95; // 5% high risk
    const isWarning = !isBypass && Math.random() > 0.85; // 10% medium risk
    
    // Strict Inland TRNC Topological Bounding Boxes to guarantee zero ocean spillage
    const r = Math.random();
    let lat, lng;
    if (r < 0.45) { 
      // Central Inland Safe-corridor (Lefkoşa to Güzelyurt) 
      lat = 35.18 + (Math.random() * 0.07); 
      lng = 32.95 + (Math.random() * 0.35); 
    } else if (r < 0.80) { 
      // Eastern Inland Plains (Geçitkale to Mağusa)
      lat = 35.15 + (Math.random() * 0.12); 
      lng = 33.40 + (Math.random() * 0.45); 
    } else { 
      // Karpaz Peninsula (Extremely thin safe line to trace the land bridge)
      lng = 34.00 + (Math.random() * 0.45); 
      lat = 35.33 + ((lng - 34.0) / 0.45) * 0.28 + (Math.random() * 0.03 - 0.015);
    }

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
