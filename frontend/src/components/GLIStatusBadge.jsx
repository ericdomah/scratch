import React from 'react';
import PropTypes from 'prop-types';

/**
 * GLIStatusBadge Component for the GridGuard AI SCADA dashboard.
 * Resolves Fix 7: Visual health indicator for the four GLI fallback modes
 * (LIVE, STALE, ESTIMATED, ABSENT) tailored for high-contrast dark UI control rooms.
 */
export const GLIStatusBadge = ({ gliStatus, stalenessMinutes = 0 }) => {
  // Determine badge styling and content based on GLI status
  const getBadgeConfig = (status) => {
    const sanitizedStatus = (status || 'ABSENT').toUpperCase();
    
    switch (sanitizedStatus) {
      case 'LIVE':
        return {
          containerClass: 'bg-emerald-950/40 text-emerald-400 border border-emerald-500/30 shadow-emerald-500/10',
          dotClass: 'bg-emerald-400 animate-pulse',
          label: 'GLI: LIVE',
          tooltip: 'Grid Load Index is fully synchronized from live substation telemetry.'
        };
      case 'STALE':
        return {
          containerClass: 'bg-amber-950/40 text-amber-400 border border-amber-500/30 shadow-amber-500/10',
          dotClass: 'bg-amber-400',
          label: `GLI: STALE (${stalenessMinutes}m)`,
          tooltip: `Delayed master meter data; using stale cache (${stalenessMinutes} minutes old).`
        };
      case 'ESTIMATED':
        return {
          containerClass: 'bg-orange-950/30 text-orange-400 border border-dashed border-orange-500/40 shadow-orange-500/5',
          dotClass: 'bg-orange-400',
          label: 'GLI: ESTIMATED',
          tooltip: 'Master meter delayed >30m; using 7-day rolling historical baseline. Spatial context degraded.'
        };
      case 'ABSENT':
      default:
        return {
          containerClass: 'bg-rose-950/50 text-rose-400 border border-rose-500/40 shadow-rose-500/20 animate-pulse',
          dotClass: 'bg-rose-500',
          label: 'GLI: ABSENT (LOW CONFIDENCE)',
          tooltip: 'No active or historical GLI data available. Feature channel set to population mean. Mandatory human review required!'
        };
    }
  };

  const config = getBadgeConfig(gliStatus);

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide border shadow-sm transition-all duration-300 select-none cursor-help ${config.containerClass}`}
      title={config.tooltip}
      id="gli-status-badge"
    >
      <span className={`w-2 h-2 rounded-full ${config.dotClass}`} />
      <span>{config.label}</span>
    </div>
  );
};

GLIStatusBadge.propTypes = {
  gliStatus: PropTypes.oneOf(['LIVE', 'STALE', 'ESTIMATED', 'ABSENT']).isRequired,
  stalenessMinutes: PropTypes.number
};

export default GLIStatusBadge;
