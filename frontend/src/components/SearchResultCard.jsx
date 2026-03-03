export default function SearchResultCard({ resource }) {
  const getTypeBadgeColor = (type) => {
    const colors = {
      'API': '#3B82F6',
      'Model': '#06b6d4',
      'Dataset': '#10b981'
    };
    return colors[type] || '#3B82F6';
  };

  const getPricingBadgeColor = (pricing) => {
    return pricing === 'free' ? '#10b981' : '#f59e0b';
  };

  return (
    <div className="glass-panel" style={{ 
      borderRadius: '16px', 
      padding: '24px', 
      cursor: 'pointer',
      transition: 'all 300ms ease',
      border: '1px solid rgba(255, 255, 255, 0.08)'
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = 'translateY(-4px)';
      e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.3)';
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <h3 style={{ 
            fontSize: '20px', 
            fontWeight: '600', 
            color: 'white', 
            marginBottom: '8px',
            lineHeight: '1.3'
          }}>
            {resource.name}
          </h3>
          
          <p style={{ 
            fontSize: '15px', 
            color: '#cbd5e1', 
            marginBottom: '16px',
            lineHeight: '1.6'
          }}>
            {resource.description}
          </p>
          
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{
              background: getTypeBadgeColor(resource.resource_type),
              color: 'white',
              padding: '6px 12px',
              borderRadius: '20px',
              fontSize: '12px',
              fontWeight: '600'
            }}>
              {resource.resource_type}
            </span>
            
            <span style={{
              background: getPricingBadgeColor(resource.pricing_type),
              color: 'white',
              padding: '6px 12px',
              borderRadius: '20px',
              fontSize: '12px',
              fontWeight: '600'
            }}>
              {resource.pricing_type.toUpperCase()}
            </span>

            {resource.github_stars && (
              <span style={{ 
                color: '#94a3b8', 
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                <span className="material-icons-round" style={{ fontSize: '14px' }}>star</span>
                {resource.github_stars.toLocaleString()}
              </span>
            )}

            {resource.downloads && (
              <span style={{ 
                color: '#94a3b8', 
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                <span className="material-icons-round" style={{ fontSize: '14px' }}>download</span>
                {resource.downloads.toLocaleString()}
              </span>
            )}
          </div>
        </div>
        
        {resource.score && (
          <div style={{
            background: 'linear-gradient(135deg, #3B82F6, #06b6d4)',
            color: 'white',
            padding: '12px 20px',
            borderRadius: '16px',
            fontWeight: '700',
            fontSize: '18px',
            textAlign: 'center',
            minWidth: '80px',
            boxShadow: '0 4px 15px rgba(59, 130, 246, 0.3)'
          }}>
            {(resource.score * 100).toFixed(0)}%
            <div style={{ fontSize: '10px', fontWeight: '500', opacity: 0.8 }}>MATCH</div>
          </div>
        )}
      </div>
    </div>
  );
}