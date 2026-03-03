import { useState, useCallback } from 'react';
import { useSearch } from '../hooks/useSearch';
import { useResources } from '../hooks/useResources';
import SearchResultCard from '../components/SearchResultCard';
import '../styles/trinity.css';

export default function TrinityDashboard() {
  console.log('🚀 TrinityDashboard rendering...');
  
  const [activeTab, setActiveTab] = useState('All');
  const [pricingMode, setPricingMode] = useState('paid');
  const [searchQuery, setSearchQuery] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([
    {
      type: 'user',
      content: 'Mujhe ek aisa image generation API chahiye jo Indian cultural elements ko accurately render kare.'
    },
    {
      type: 'ai',
      content: 'Maine aapke liye kuch models shortlist kiye hain jo Indian aesthetics pe fine-tune kiye gaye hain.',
      suggestions: [
        { name: 'Kling 3.0 (Indian LoRA)', description: 'Optimized for sarees, festivals, and architecture.' },
        { name: 'DesiDiffuser V2', description: 'High fidelity portraits with Indian skin tones.' }
      ]
    }
  ]);

  const { results, loading, search, clearResults } = useSearch();
  const { resources: apiResources } = useResources('API');
  const { resources: modelResources } = useResources('Model');
  const { resources: datasetResources } = useResources('Dataset');

  console.log('📊 Resources loaded:', { apiResources, modelResources, datasetResources });

  const handleSearch = useCallback(async (query) => {
    if (!query.trim()) return;
    
    const filters = {
      pricing_filter: pricingMode === 'both' ? null : [pricingMode],
      resource_types: activeTab === 'All' ? null : [activeTab],
      limit: 20
    };
    
    await search(query, filters);
  }, [search, pricingMode, activeTab]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch(searchQuery);
    }
  };

  const handleChatSubmit = () => {
    if (!chatInput.trim()) return;
    
    setChatMessages(prev => [...prev, {
      type: 'user',
      content: chatInput
    }]);
    
    // Simulate AI response
    setTimeout(() => {
      setChatMessages(prev => [...prev, {
        type: 'ai',
        content: 'Let me help you find the right resources for that requirement.'
      }]);
    }, 1000);
    
    setChatInput('');
  };

  return (
    <>
      {/* Gradient Orbs */}
      <div className="gradient-orb orb-1"></div>
      <div className="gradient-orb orb-2"></div>
      <div className="gradient-orb orb-3"></div>

      <div style={{ maxWidth: '1600px', margin: '0 auto', padding: '24px', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <header style={{ display: 'flex', flexDirection: 'column', gap: '24px', marginBottom: '32px' }}>
          {/* Navigation */}
          <nav className="glass-panel" style={{ borderRadius: '16px', padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ width: '40px', height: '40px', background: 'linear-gradient(135deg, #3B82F6, #A855F7)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(59, 130, 246, 0.2)' }}>
                <span className="material-icons-round" style={{ color: 'white', fontSize: '20px' }}>code</span>
              </div>
              <span style={{ fontSize: '20px', fontWeight: '700', color: 'white' }}>DevStore</span>
            </div>

            {/* Tab Navigation */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(30, 41, 59, 0.5)', padding: '6px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
              {['All', 'APIs', 'Models', 'Datasets', 'Blueprints'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  style={{
                    padding: '6px 16px',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '500',
                    background: activeTab === tab ? '#1e293b' : 'transparent',
                    color: activeTab === tab ? 'white' : '#94a3b8',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'all 200ms',
                    boxShadow: activeTab === tab ? '0 2px 8px rgba(0,0,0,0.2)' : 'none'
                  }}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Right Actions */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 12px', borderRadius: '8px', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.2)' }}>
                <span style={{ position: 'relative', display: 'flex', width: '8px', height: '8px' }}>
                  <span className="animate-ping" style={{ position: 'absolute', display: 'inline-flex', height: '100%', width: '100%', borderRadius: '50%', background: '#22c55e', opacity: 0.75 }}></span>
                  <span style={{ position: 'relative', display: 'inline-flex', borderRadius: '50%', height: '8px', width: '8px', background: '#22c55e' }}></span>
                </span>
                <span style={{ fontSize: '12px', fontFamily: 'JetBrains Mono, monospace', color: '#22c55e' }}>Hinglish Mode</span>
              </div>
              <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: '#1e293b', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                <span className="material-icons-round" style={{ fontSize: '16px', color: '#94a3b8' }}>notifications</span>
              </div>
              <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, #ec4899, #f43f5e)', border: '2px solid #020617' }}></div>
            </div>
          </nav>

          {/* Search Bar */}
          <div className="glass-panel" style={{ borderRadius: '16px', padding: '24px', display: 'flex', alignItems: 'center', gap: '24px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderRight: '1px solid rgba(255, 255, 255, 0.1)', paddingRight: '24px' }}>
              <span style={{ fontSize: '14px', fontWeight: '500', color: '#94a3b8' }}>Free</span>
              <button
                onClick={() => setPricingMode(pricingMode === 'free' ? 'paid' : 'free')}
                style={{
                  width: '48px',
                  height: '24px',
                  background: '#3B82F6',
                  borderRadius: '12px',
                  position: 'relative',
                  border: 'none',
                  cursor: 'pointer',
                  boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)'
                }}
              >
                <span style={{
                  position: 'absolute',
                  right: pricingMode === 'paid' ? '4px' : 'auto',
                  left: pricingMode === 'free' ? '4px' : 'auto',
                  top: '4px',
                  width: '16px',
                  height: '16px',
                  background: 'white',
                  borderRadius: '50%',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                  transition: 'all 200ms'
                }}></span>
              </button>
              <span style={{ fontSize: '14px', fontWeight: '500', color: 'white' }}>Paid</span>
            </div>

            <div style={{ flex: 1, position: 'relative', minWidth: '300px' }}>
              <span className="material-icons-round" style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8', fontSize: '20px' }}>search</span>
              <input
                type="text"
                placeholder="Search for 'Image Generation API', 'Llama 3', or 'Customer Data'..."
                style={{
                  width: '100%',
                  background: 'rgba(15, 23, 42, 0.5)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                  padding: '16px 16px 16px 48px',
                  color: 'white',
                  fontSize: '15px',
                  outline: 'none',
                  transition: 'all 200ms'
                }}
                onFocus={(e) => e.target.style.borderColor = '#3B82F6'}
                onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
              />
              <div style={{ position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', display: 'flex', gap: '8px' }}>
                <span style={{ padding: '4px 8px', background: '#1e293b', borderRadius: '4px', fontSize: '12px', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>⌘ K</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Grid */}
        <main style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '24px', gridAutoRows: 'minmax(180px, auto)' }}>
          {/* Intent Discovery Chat - Large */}
          <IntentDiscovery
            messages={chatMessages}
            input={chatInput}
            setInput={setChatInput}
            onSubmit={handleChatSubmit}
          />

          {/* Solution Blueprint - Large */}
          <SolutionBlueprint />

          {/* Top APIs */}
          <ResourceColumn
            title="Top APIs"
            icon="code"
            color="blue"
            resources={[
              { name: 'Azure OpenAI', version: 'v4.0', rating: 4.9, pricing: 'Paid', initial: 'A', bgColor: '#1e3a8a' },
              { name: 'Stable Video', version: 'v1.1', rating: 4.7, pricing: 'Freemium', initial: 'S', bgColor: '#374151' }
            ]}
            buttonText="1-Click Boilerplate"
          />

          {/* Optimized Models */}
          <ResourceColumn
            title="Optimized Models"
            icon="psychology"
            color="cyan"
            resources={[
              { name: 'Llama-3-70b', version: 'Quantized', downloads: '2.1M', initial: 'L', bgColor: '#1e40af', hasImage: true },
              { name: 'Mistral Large', version: 'Instruct', downloads: '800k', initial: 'M', bgColor: '#4338ca' }
            ]}
            buttonText="Deploy Serverless"
          />

          {/* Popular Datasets */}
          <ResourceColumn
            title="Popular Datasets"
            icon="storage"
            color="emerald"
            resources={[
              { name: 'Common Crawl (Hi)', size: '500GB', tag: 'Cleaned', icon: 'storage', bgColor: '#065f46' },
              { name: 'LAION Indian Art', size: '12GB', tag: 'Tagged', icon: 'image', bgColor: '#065f46' }
            ]}
            buttonText="Mount Volume"
          />
        </main>
      </div>
    </>
  );
}

// Intent Discovery Component
function IntentDiscovery({ messages, input, setInput, onSubmit }) {
  return (
    <div className="glass-panel" style={{ gridColumn: 'span 12 / span 12', gridRow: 'span 2', borderRadius: '16px', padding: '24px', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '18px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="material-icons-round" style={{ color: '#a78bfa' }}>psychology</span>
          Intent Discovery
        </h3>
        <span style={{ padding: '4px 8px', borderRadius: '4px', background: 'rgba(168, 85, 247, 0.1)', color: '#a78bfa', fontSize: '12px', fontFamily: 'JetBrains Mono, monospace', border: '1px solid rgba(168, 85, 247, 0.2)' }}>RAG Enabled</span>
      </div>

      <div className="custom-scrollbar" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', paddingRight: '8px' }}>
        {messages.map((message, idx) => (
          <div key={idx} style={{ display: 'flex', gap: '12px', justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start' }}>
            {message.type === 'user' ? (
              <>
                <div style={{ background: '#3B82F6', color: 'white', padding: '16px', borderRadius: '16px 16px 4px 16px', maxWidth: '80%', boxShadow: '0 4px 15px rgba(59, 130, 246, 0.1)' }}>
                  <p style={{ fontSize: '14px', lineHeight: '1.6' }}>{message.content}</p>
                </div>
                <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: '#1e293b', flexShrink: 0 }}></div>
              </>
            ) : (
              <>
                <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'linear-gradient(135deg, #3B82F6, #A855F7)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <span className="material-icons-round" style={{ color: 'white', fontSize: '16px' }}>smart_toy</span>
                </div>
                <div style={{ background: 'rgba(30, 41, 59, 0.8)', padding: '16px', borderRadius: '16px 16px 16px 4px', maxWidth: '90%', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: '#cbd5e1', marginBottom: message.suggestions ? '16px' : '0' }}>
                    <span style={{ color: 'white', fontWeight: '500' }}>Suggestion:</span> {message.content}
                  </p>
                  {message.suggestions && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                      {message.suggestions.map((suggestion, sidx) => (
                        <ModelCard key={sidx} name={suggestion.name} description={suggestion.description} />
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Input */}
      <div style={{ marginTop: '16px', position: 'relative' }}>
        <input
          type="text"
          placeholder="Ask follow up..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && onSubmit()}
          style={{
            width: '100%',
            background: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '12px 48px 12px 16px',
            fontSize: '14px',
            color: 'white',
            outline: 'none'
          }}
        />
        <button 
          onClick={onSubmit}
          style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', padding: '6px', background: '#a855f7', borderRadius: '8px', color: 'white', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span className="material-icons-round" style={{ fontSize: '16px' }}>send</span>
        </button>
      </div>
    </div>
  );
}

function ModelCard({ name, description }) {
  return (
    <div style={{ background: '#0f172a', padding: '12px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.1)', cursor: 'pointer', transition: 'all 200ms' }}
      onMouseEnter={(e) => e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.5)'}
      onMouseLeave={(e) => e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)'}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span style={{ fontSize: '12px', fontWeight: '700', color: 'white' }}>{name}</span>
        <span className="material-icons-round" style={{ fontSize: '12px', color: '#3B82F6' }}>arrow_forward</span>
      </div>
      <p style={{ fontSize: '10px', color: '#94a3b8', lineHeight: '1.4' }} className="line-clamp-2">{description}</p>
    </div>
  );
}

// Solution Blueprint Component
function SolutionBlueprint() {
  return (
    <div className="glass-panel" style={{ gridColumn: 'span 12 / span 12', gridRow: 'span 2', borderRadius: '16px', padding: '24px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '18px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="material-icons-round" style={{ color: '#06b6d4' }}>schema</span>
          Solution Blueprint
        </h3>
        <button style={{ fontSize: '12px', color: '#94a3b8', background: 'none', border: 'none', cursor: 'pointer' }}>View Full</button>
      </div>

      <div style={{ flex: 1, background: 'rgba(15, 23, 42, 0.5)', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)', position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}>
        <div className="dot-pattern" style={{ position: 'absolute', inset: 0 }}></div>
        
        <div style={{ position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', gap: '32px' }}>
          <BlueprintNode icon="dataset" label="Input Data" color="#06b6d4" />
          <Arrow />
          <BlueprintNode icon="auto_awesome" label="LLM Process" color="#a855f7" />
          <Arrow />
          <BlueprintNode icon="api" label="API Output" color="#22c55e" />
        </div>
      </div>

      <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', color: '#94a3b8' }}>Retrieval Augmented Generation Flow</span>
        <button style={{ padding: '6px 12px', background: 'rgba(6, 182, 212, 0.1)', color: '#06b6d4', fontSize: '12px', fontWeight: '500', borderRadius: '8px', border: '1px solid rgba(6, 182, 212, 0.2)', cursor: 'pointer' }}>Use Template</button>
      </div>
    </div>
  );
}

function BlueprintNode({ icon, label, color }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
      <div style={{ width: '64px', height: '64px', background: '#1e293b', borderRadius: '12px', border: `1px solid ${color}80`, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `0 0 15px ${color}4D` }}>
        <span className="material-icons-round" style={{ color, fontSize: '32px' }}>{icon}</span>
      </div>
      <span style={{ fontSize: '12px', fontFamily: 'JetBrains Mono, monospace', color: '#94a3b8' }}>{label}</span>
    </div>
  );
}

function Arrow() {
  return (
    <div style={{ position: 'relative', width: '48px', height: '2px', background: '#374151' }}>
      <div style={{ position: 'absolute', right: 0, top: '-4px', width: '8px', height: '8px', borderTop: '2px solid #64748b', borderRight: '2px solid #64748b', transform: 'rotate(45deg)' }}></div>
      <div className="animate-pulse" style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '8px', height: '8px', background: '#3B82F6', borderRadius: '50%' }}></div>
    </div>
  );
}

// Resource Column Component
function ResourceColumn({ title, icon, color, resources, buttonText }) {
  const colors = {
    blue: { dot: '#3B82F6', text: '#60a5fa', border: '#1e40af' },
    cyan: { dot: '#06b6d4', text: '#22d3ee', border: '#0e7490' },
    emerald: { dot: '#10b981', text: '#34d399', border: '#047857' }
  };

  return (
    <div className="glass-panel" style={{ gridColumn: 'span 12 / span 12', borderRadius: '16px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '12px' }}>
        <h4 style={{ fontWeight: '500', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: colors[color].dot, boxShadow: `0 0 8px ${colors[color].dot}` }}></span>
          {title}
        </h4>
        <a href="#" style={{ fontSize: '12px', color: colors[color].text, textDecoration: 'none' }}>View All</a>
      </div>

      {resources.slice(0, 2).map((resource, idx) => (
        <ResourceCard key={idx} resource={resource} buttonText={buttonText} color={color} />
      ))}
    </div>
  );
}

function ResourceCard({ resource, buttonText, color }) {
  const colors = {
    blue: { button: '#3B82F6', buttonBg: 'rgba(59, 130, 246, 0.1)', buttonBorder: 'rgba(59, 130, 246, 0.2)' },
    cyan: { button: '#06b6d4', buttonBg: 'rgba(6, 182, 212, 0.1)', buttonBorder: 'rgba(6, 182, 212, 0.2)' },
    emerald: { button: '#10b981', buttonBg: 'rgba(16, 185, 129, 0.1)', buttonBorder: 'rgba(16, 185, 129, 0.2)' }
  };

  return (
    <div style={{ padding: '12px', borderRadius: '12px', background: 'rgba(30, 41, 59, 0.4)', border: '1px solid rgba(255, 255, 255, 0.05)', cursor: 'pointer', transition: 'all 200ms' }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'rgba(30, 41, 59, 0.6)';
        e.currentTarget.style.borderColor = colors[color].buttonBorder;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'rgba(30, 41, 59, 0.4)';
        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.05)';
      }}>
      <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {resource.icon ? (
            <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: '#065f46', display: 'flex', alignItems: 'center', justifyContent: 'center', border: `1px solid ${colors[color].buttonBorder}` }}>
              <span className="material-icons-round" style={{ color: colors[color].button, fontSize: '18px' }}>{resource.icon}</span>
            </div>
          ) : (
            <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: '#1e40af', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: '700', border: `1px solid ${colors[color].buttonBorder}` }}>
              {resource.initial}
            </div>
          )}
          <div>
            <h5 style={{ fontSize: '14px', fontWeight: '600', color: '#e2e8f0', marginBottom: '4px' }}>{resource.name}</h5>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '10px', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>{resource.version || resource.size}</span>
              {resource.rating && (
                <span style={{ fontSize: '10px', color: '#eab308', display: 'flex', alignItems: 'center', gap: '2px' }}>
                  <span className="material-icons-round" style={{ fontSize: '10px' }}>star</span> {resource.rating}
                </span>
              )}
              {resource.downloads && (
                <span style={{ fontSize: '10px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '2px' }}>
                  <span className="material-icons-round" style={{ fontSize: '10px' }}>download</span> {resource.downloads}
                </span>
              )}
              {resource.tag && (
                <span style={{ fontSize: '10px', background: colors[color].buttonBg, color: colors[color].button, padding: '2px 6px', borderRadius: '4px' }}>{resource.tag}</span>
              )}
            </div>
          </div>
        </div>
        {resource.pricing_type && (
          <span style={{ fontSize: '10px', background: 'rgba(51, 65, 85, 0.5)', padding: '4px 8px', borderRadius: '4px', color: '#94a3b8' }}>{resource.pricing_type}</span>
        )}
      </div>
      <button style={{ width: '100%', padding: '8px', borderRadius: '8px', background: colors[color].buttonBg, color: colors[color].button, fontSize: '12px', fontWeight: '500', border: `1px solid ${colors[color].buttonBorder}`, cursor: 'pointer', transition: 'all 200ms' }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = colors[color].button;
          e.currentTarget.style.color = 'white';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = colors[color].buttonBg;
          e.currentTarget.style.color = colors[color].button;
        }}>
        {buttonText}
      </button>
    </div>
  );
}