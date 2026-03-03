import './ResourceCard.css';

export default function ResourceCard({ resource }) {
  const getTypeBadgeClass = (type) => {
    const typeMap = {
      'API': 'badge-api',
      'Model': 'badge-model',
      'Dataset': 'badge-dataset'
    };
    return typeMap[type] || 'badge-api';
  };

  return (
    <div className="resource-card glass-card">
      <div className="resource-card-content">
        <div className="resource-card-main">
          <h3 className="resource-card-title">{resource.name}</h3>
          <p className="resource-card-description">{resource.description}</p>
          
          <div className="resource-card-badges">
            <span className={`badge ${getTypeBadgeClass(resource.resource_type)}`}>
              {resource.resource_type}
            </span>
            <span className={`badge ${resource.pricing_type === 'free' ? 'badge-free' : 'badge-paid'}`}>
              {resource.pricing_type.toUpperCase()}
            </span>
          </div>
        </div>
        
        {resource.score && (
          <div className="resource-card-score">
            <div className="score-badge">
              {(resource.score * 100).toFixed(0)}%
            </div>
            <span className="score-label">Match</span>
          </div>
        )}
      </div>
    </div>
  );
}
