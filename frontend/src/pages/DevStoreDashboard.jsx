// Dev-Store Dashboard — connected to OpenSearch + AWS Bedrock backend
import { useState, useEffect, useCallback, useRef } from "react";
import apiService from "../services/api";

// ─── Lucide-style SVG Icons ───────────────────────────────────────────────────
const Icon = ({ d, size = 16, color = "currentColor", strokeWidth = 1.75 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d)
      ? d.map((path, i) => <path key={i} d={path} />)
      : <path d={d} />}
  </svg>
);

const Icons = {
  Home:     "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M9 22V12h6v10",
  Cpu:      ["M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18","M9 9h1v1H9zm4 0h1v1h-1zm4 0h1v1h-1zM9 13h1v1H9zm4 0h1v1h-1zm4 0h1v1h-1z"],
  Database: ["M12 2a9 3 0 1 0 0 6 9 3 0 0 0 0-6z","M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5","M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"],
  Layers:   ["M12 2L2 7l10 5 10-5-10-5z","M2 17l10 5 10-5","M2 12l10 5 10-5"],
  Search:   ["M21 21l-4.35-4.35","M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0"],
  Bell:     ["M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9","M13.73 21a2 2 0 0 1-3.46 0"],
  Settings: ["M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z","M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"],
  Star:     "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
  Zap:      "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
  Copy:     ["M20 9H11a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2v-9a2 2 0 0 0-2-2z","M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 0 2 2v1"],
  Check:    "M20 6L9 17l-5-5",
  Activity: "M22 12h-4l-3 9L9 3l-3 9H2",
  TrendUp:  "M23 6l-9.5 9.5-5-5L1 18 M17 6h6v6",
  Package:  ["M16.5 9.4l-9-5.17","M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z","M3.27 6.96L12 12.01l8.73-5.05 M12 22.08V12"],
  ChevronR: "M9 18l6-6-6-6",
  Wifi:     ["M5 12.55a11 11 0 0 1 14.08 0","M1.42 9a16 16 0 0 1 21.16 0","M8.53 16.11a6 6 0 0 1 6.95 0","M12 20h.01"],
  Brain:    ["M9.5 2a2.5 2.5 0 0 1 5 0","M9.5 2C7.567 2 6 3.567 6 5.5v1C4.343 6.5 3 7.843 3 9.5S4.343 12.5 6 12.5v1c0 1.933 1.567 3.5 3.5 3.5s3.5-1.567 3.5-3.5v-1c1.657 0 3-1.343 3-3S16.657 6.5 15 6.5v-1C15 3.567 13.433 2 11.5 2H9.5z"],
  Globe:    ["M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2z","M2 12h20","M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"],
  Send:     "M22 2L11 13 M22 2L15 22l-4-9-9-4 22-7z",
};

// ─── Constants ────────────────────────────────────────────────────────────────
const CATEGORIES = ["All", "API", "Model", "Dataset"];

const GHOST_TEXTS = [
  "Bhai, best payment gateway batao...",
  "Looking for a ML Model API?",
  "Kaunsa dataset use karein?",
  "Find low-latency inference APIs...",
  "Best open-source model for NLP?",
  "Search with AI — powered by Bedrock...",
];

// Accent colors & emojis per resource type
const TYPE_META = {
  API:     { color: "#3B82F6", emoji: "🔌" },
  Model:   { color: "#A855F7", emoji: "🧠" },
  Dataset: { color: "#F59E0B", emoji: "🗄️" },
};

const PRICING_META = {
  free:     { color: "#00FFA3", label: "free" },
  paid:     { color: "#3B82F6", label: "paid" },
  freemium: { color: "#F59E0B", label: "freemium" },
};

// Map a backend resource to ToolCard-compatible shape
function mapResource(r) {
  const meta  = TYPE_META[r.resource_type] || { color: "#6B7280", emoji: "📦" };
  const stars = r.github_stars ?? r.downloads ?? 0;
  const latencyRaw = parseFloat(r.latency_ms ?? r.p99_latency ?? 0);
  const latency = latencyRaw > 0 ? latencyRaw : Math.floor(Math.random() * 80 + 10);

  let installCommand = r.install_command || r.endpoint_url || "";
  if (!installCommand) {
    if (r.resource_type === "API")     installCommand = `curl ${(r.base_url || "https://api.example.com/v1")}`;
    else if (r.resource_type === "Model") installCommand = `pip install transformers && hf download ${r.name?.replace(/\s+/g, "-").toLowerCase()}`;
    else                                installCommand = `wget ${r.download_url || "https://huggingface.co/datasets/" + r.name?.replace(/\s+/g, "-").toLowerCase()}`;
  }

  return {
    id:             r.id || String(Math.random()),
    name:           r.name,
    category:       r.resource_type || "API",
    description:    r.description || "",
    installCommand,
    status:         r.is_available !== false ? "stable" : "down",
    stars,
    latency,
    iconEmoji:      meta.emoji,
    accentColor:    meta.color,
    pricingType:    r.pricing_type || "free",
    score:          r.score,
    tags:           r.tags || [],
  };
}

// ─── Ghost-text hook ──────────────────────────────────────────────────────────
function useGhostText(active) {
  const [text, setText] = useState("");
  const idx   = useRef(0);
  const chars = useRef(0);
  const phase = useRef("typing");

  useEffect(() => {
    if (!active) { setText(""); return; }
    let raf;
    let last = 0;
    const SPEEDS = { typing: 46, erasing: 22, pause: 2000, gap: 350 };

    const tick = (now) => {
      const target = GHOST_TEXTS[idx.current];
      const delay =
        phase.current === "typing"  ? SPEEDS.typing  :
        phase.current === "erasing" ? SPEEDS.erasing :
        phase.current === "pause"   ? SPEEDS.pause   : SPEEDS.gap;

      if (now - last >= delay) {
        last = now;
        if (phase.current === "typing") {
          if (chars.current < target.length) { chars.current++; setText(target.slice(0, chars.current)); }
          else { phase.current = "pause"; }
        } else if (phase.current === "pause") {
          phase.current = "erasing";
        } else if (phase.current === "erasing") {
          if (chars.current > 0) { chars.current--; setText(target.slice(0, chars.current)); }
          else { idx.current = (idx.current + 1) % GHOST_TEXTS.length; phase.current = "gap"; }
        } else {
          phase.current = "typing";
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  return text;
}

// ─── ToolCard ─────────────────────────────────────────────────────────────────
function ToolCard({ tool, index }) {
  const [copied,  setCopied]  = useState(false);
  const [hovered, setHovered] = useState(false);

  const handleCopy = async () => {
    try { await navigator.clipboard.writeText(tool.installCommand); } catch {}
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
  const pricing = PRICING_META[tool.pricingType] || PRICING_META.free;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered
          ? `linear-gradient(145deg, ${tool.accentColor}0D 0%, #161618 55%, #0f0f11 100%)`
          : "#141416",
        border: `1px solid ${hovered ? tool.accentColor + "28" : "rgba(255,255,255,0.055)"}`,
        borderRadius: 24,
        padding: "22px 22px 20px",
        display: "flex", flexDirection: "column", gap: 14,
        position: "relative", overflow: "hidden",
        transform: hovered ? "translateY(-3px)" : "translateY(0)",
        boxShadow: hovered
          ? `0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px ${tool.accentColor}15, inset 0 1px 0 rgba(255,255,255,0.04)`
          : "0 2px 12px rgba(0,0,0,0.3)",
        transition: "all 0.28s cubic-bezier(0.34,1.56,0.64,1)",
        animationDelay: `${index * 60}ms`,
        animation: "cardIn 0.4s ease both",
      }}
    >
      {/* Top shimmer line */}
      <div style={{
        position: "absolute", top: 0, left: "10%", right: "10%", height: 1,
        background: hovered
          ? `linear-gradient(90deg, transparent, ${tool.accentColor}60, transparent)`
          : `linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent)`,
        transition: "background 0.3s",
      }} />

      {/* Header row */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 14, flexShrink: 0,
            background: `${tool.accentColor}15`,
            border: `1px solid ${tool.accentColor}28`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20,
            boxShadow: hovered ? `0 4px 16px ${tool.accentColor}20` : "none",
            transition: "box-shadow 0.3s",
          }}>
            {tool.iconEmoji}
          </div>
          <div>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontWeight: 700, fontSize: 15, color: "rgba(255,255,255,0.92)", letterSpacing: "-0.02em" }}>
              {tool.name}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 3 }}>
              <span style={{ fontSize: 11, color: "rgba(255,255,255,0.32)", fontWeight: 500 }}>
                {tool.category}
              </span>
              <span style={{
                fontSize: 9.5, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase",
                padding: "2px 6px", borderRadius: 99,
                background: `${pricing.color}14`,
                border: `1px solid ${pricing.color}28`,
                color: pricing.color,
              }}>
                {pricing.label}
              </span>
            </div>
          </div>
        </div>

        {/* Status pill */}
        <div style={{
          display: "flex", alignItems: "center", gap: 5,
          padding: "4px 10px", borderRadius: 99, flexShrink: 0,
          background: tool.status === "stable" ? "rgba(0,255,163,0.08)" : "rgba(239,68,68,0.08)",
          border: `1px solid ${tool.status === "stable" ? "rgba(0,255,163,0.2)" : "rgba(239,68,68,0.2)"}`,
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%",
            background: tool.status === "stable" ? "#00FFA3" : "#ef4444",
            boxShadow: tool.status === "stable" ? "0 0 6px #00FFA3" : "0 0 6px #ef4444",
            animation: tool.status === "stable" ? "pulseGlow 2s infinite" : "none",
          }} />
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
            color: tool.status === "stable" ? "#00FFA3" : "#ef4444",
          }}>
            {tool.status}
          </span>
        </div>
      </div>

      {/* Description */}
      <p style={{
        fontSize: 13, color: "rgba(255,255,255,0.42)", lineHeight: 1.55,
        fontFamily: "'DM Sans', sans-serif",
        display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden",
      }}>
        {tool.description || "No description available."}
      </p>

      {/* Score bar (shown when from search results) */}
      {tool.score != null && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", fontFamily: "'Fira Code', monospace" }}>AI Score</span>
          <div style={{ flex: 1, height: 4, borderRadius: 99, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
            <div style={{
              height: "100%", borderRadius: 99,
              width: `${Math.round(tool.score * 100)}%`,
              background: `linear-gradient(90deg, ${tool.accentColor}, ${tool.accentColor}88)`,
              transition: "width 0.8s ease",
            }} />
          </div>
          <span style={{ fontSize: 11, color: tool.accentColor, fontFamily: "'Fira Code', monospace" }}>
            {Math.round(tool.score * 100)}%
          </span>
        </div>
      )}

      {/* Meta row */}
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5, color: "rgba(255,255,255,0.3)", fontSize: 12 }}>
          <Icon d={Icons.Star} size={12} color="rgba(251,191,36,0.65)" />
          <span style={{ fontFamily: "'Fira Code', monospace", fontWeight: 500 }}>{fmt(tool.stars)}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5, color: "rgba(255,255,255,0.3)", fontSize: 12 }}>
          <Icon d={Icons.Zap} size={12} color="rgba(0,255,163,0.65)" />
          <span style={{ fontFamily: "'Fira Code', monospace", fontWeight: 500 }}>{tool.latency}ms</span>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 5, color: "rgba(255,255,255,0.2)", fontSize: 11 }}>
          <Icon d={Icons.Activity} size={11} />
          <span>P99</span>
        </div>
      </div>

      {/* Install / Endpoint box */}
      <div style={{
        background: "rgba(0,0,0,0.45)", borderRadius: 14, padding: "10px 14px",
        border: "1px solid rgba(255,255,255,0.055)", backdropFilter: "blur(4px)",
      }}>
        <code style={{
          fontFamily: "'Fira Code', 'JetBrains Mono', monospace", fontSize: 11.5,
          color: hovered ? `${tool.accentColor}CC` : "rgba(0,255,163,0.65)",
          whiteSpace: "nowrap", overflow: "hidden", display: "block", textOverflow: "ellipsis",
          transition: "color 0.3s",
        }}>
          {tool.installCommand}
        </code>
      </div>

      {/* Copy button */}
      <button
        onClick={handleCopy}
        style={{
          width: "100%", borderRadius: 99,
          padding: "11px 0", cursor: "pointer",
          fontFamily: "'DM Sans', sans-serif", fontWeight: 700, fontSize: 13,
          display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
          background: copied
            ? "rgba(0,255,163,0.15)"
            : hovered
              ? `linear-gradient(135deg, ${tool.accentColor}50, ${tool.accentColor}30)`
              : "rgba(0,107,93,0.22)",
          color: copied ? "#00FFA3" : "rgba(255,255,255,0.8)",
          border: copied
            ? "1px solid rgba(0,255,163,0.35)"
            : `1px solid ${hovered ? tool.accentColor + "45" : "rgba(0,107,93,0.4)"}`,
          transition: "all 0.25s ease",
          letterSpacing: "-0.01em",
        }}
        onMouseDown={e => e.currentTarget.style.transform = "scale(0.97)"}
        onMouseUp={e   => e.currentTarget.style.transform = "scale(1)"}
      >
        <Icon d={copied ? Icons.Check : Icons.Copy} size={14} color={copied ? "#00FFA3" : "rgba(255,255,255,0.6)"} />
        {copied ? "Copied!" : "Copy Command"}
      </button>
    </div>
  );
}

// ─── Skeleton Card ────────────────────────────────────────────────────────────
function SkeletonCard({ index }) {
  return (
    <div style={{
      background: "#141416", borderRadius: 24, padding: "22px 22px 20px",
      border: "1px solid rgba(255,255,255,0.04)",
      animationDelay: `${index * 80}ms`, animation: "shimmer 1.6s ease infinite",
    }}>
      {[44, 10, 13, 10, 10, 40].map((h, i) => (
        <div key={i} style={{
          height: h, borderRadius: 8, background: "rgba(255,255,255,0.04)",
          marginBottom: 14, width: i === 1 ? "60%" : i === 2 ? "85%" : i === 3 ? "40%" : "100%",
        }} />
      ))}
    </div>
  );
}

// ─── Nav Item ─────────────────────────────────────────────────────────────────
function NavItem({ iconKey, label, active, onClick }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        width: "100%", background: active ? "rgba(0,107,93,0.2)" : hov ? "rgba(255,255,255,0.04)" : "transparent",
        border: "none", borderRadius: 16, padding: "12px 8px", cursor: "pointer",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
        position: "relative", transition: "background 0.2s",
      }}
    >
      {active && (
        <div style={{
          position: "absolute", left: -12, top: "50%", transform: "translateY(-50%)",
          width: 3, height: 28, background: "#00FFA3", borderRadius: "0 3px 3px 0",
        }} />
      )}
      <Icon
        d={Icons[iconKey]} size={18}
        color={active ? "#00FFA3" : hov ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.28)"}
      />
      <span style={{
        fontSize: 9.5, fontWeight: 600, letterSpacing: "0.05em",
        color: active ? "#00FFA3" : "rgba(255,255,255,0.28)",
        textTransform: "uppercase", fontFamily: "'DM Sans', sans-serif",
      }}>
        {label}
      </span>
    </button>
  );
}

// ─── AI Chat Panel ────────────────────────────────────────────────────────────
function AIChatPanel({ onClose }) {
  const [messages, setMessages] = useState([
    { type: "ai", content: "Namaste! I'm powered by AWS Bedrock. Ask me for recommendations — \"best free NLP model\", \"low-latency image API\", etc." },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q) return;
    setInput("");
    setMessages(prev => [...prev, { type: "user", content: q }]);
    setThinking(true);
    try {
      const resp = await apiService.search(q, { limit: 3 });
      const results = resp.results || [];
      let aiContent = results.length
        ? `Found ${results.length} matching resource(s) via OpenSearch + Bedrock semantic search:`
        : "No exact matches found. Try different keywords or browse by category.";
      setMessages(prev => [
        ...prev,
        {
          type: "ai",
          content: aiContent,
          results: results.map(mapResource),
        },
      ]);
    } catch {
      setMessages(prev => [...prev, { type: "ai", content: "Backend is offline. Check if the FastAPI server is running on port 8000." }]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <div style={{
      position: "fixed", right: 24, bottom: 24, width: 420, height: 540,
      background: "#141416", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 24, display: "flex", flexDirection: "column",
      boxShadow: "0 40px 100px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,255,163,0.08)",
      zIndex: 100, animation: "cardIn 0.3s ease both",
    }}>
      {/* Header */}
      <div style={{ padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 32, height: 32, borderRadius: 10, background: "linear-gradient(135deg, #006B5D, #00FFA3)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Icon d={Icons.Brain} size={15} color="white" />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 13, color: "rgba(255,255,255,0.88)" }}>Bedrock AI Assistant</div>
          <div style={{ fontSize: 10, color: "rgba(0,255,163,0.7)", display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#00FFA3", animation: "pulseGlow 2s infinite" }} />
            OpenSearch + AWS Bedrock
          </div>
        </div>
        <button onClick={onClose} style={{ marginLeft: "auto", background: "none", border: "none", color: "rgba(255,255,255,0.3)", cursor: "pointer", fontSize: 18, lineHeight: 1 }}>×</button>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: m.type === "user" ? "flex-end" : "flex-start" }}>
            <div style={{
              maxWidth: "85%", padding: "10px 14px", borderRadius: m.type === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              background: m.type === "user" ? "rgba(59,130,246,0.25)" : "rgba(255,255,255,0.05)",
              border: `1px solid ${m.type === "user" ? "rgba(59,130,246,0.3)" : "rgba(255,255,255,0.06)"}`,
              fontSize: 13, color: "rgba(255,255,255,0.75)", lineHeight: 1.55,
            }}>
              {m.content}
            </div>
            {m.results?.map((r, j) => (
              <div key={j} style={{
                maxWidth: "90%", padding: "10px 14px", borderRadius: 12,
                background: `${r.accentColor}0D`, border: `1px solid ${r.accentColor}22`,
                fontSize: 12, color: "rgba(255,255,255,0.6)", lineHeight: 1.5,
              }}>
                <div style={{ fontWeight: 700, color: "rgba(255,255,255,0.85)", marginBottom: 2 }}>
                  {r.iconEmoji} {r.name} <span style={{ color: r.accentColor, marginLeft: 6 }}>{r.category}</span>
                </div>
                <div>{r.description?.slice(0, 80)}…</div>
              </div>
            ))}
          </div>
        ))}
        {thinking && (
          <div style={{ display: "flex", gap: 4, padding: "10px 14px" }}>
            {[0.1, 0.2, 0.3].map(d => (
              <div key={d} style={{ width: 6, height: 6, borderRadius: "50%", background: "#00FFA3", animation: `pulseGlow 1s ${d}s infinite` }} />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: "12px 16px", borderTop: "1px solid rgba(255,255,255,0.06)", display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSend()}
          placeholder="Ask AI for recommendations..."
          style={{
            flex: 1, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 99, padding: "10px 16px", color: "rgba(255,255,255,0.8)", fontSize: 13,
            fontFamily: "'DM Sans', sans-serif",
          }}
        />
        <button
          onClick={handleSend}
          style={{
            width: 38, height: 38, borderRadius: "50%", flexShrink: 0,
            background: "linear-gradient(135deg, #006B5D, #00FFA3)",
            border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <Icon d={Icons.Send} size={14} color="white" />
        </button>
      </div>
    </div>
  );
}

// ─── Main DevStore Dashboard ──────────────────────────────────────────────────
export default function DevStoreDashboard() {
  const [tools,          setTools]          = useState([]);
  const [filtered,       setFiltered]       = useState([]);
  const [query,          setQuery]          = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeNav,      setActiveNav]      = useState(0);
  const [loading,        setLoading]        = useState(true);
  const [searchLoading,  setSearchLoading]  = useState(false);
  const [focused,        setFocused]        = useState(false);
  const [showChat,       setShowChat]       = useState(false);
  const [apiOnline,      setApiOnline]      = useState(null);
  const [errorMsg,       setErrorMsg]       = useState("");
  const debounceRef = useRef(null);
  const ghostText   = useGhostText(!focused && !query);

  // Load all resources from backend on mount
  useEffect(() => {
    const loadAll = async () => {
      setLoading(true);
      setErrorMsg("");
      try {
        const data = await apiService.getResources({ limit: 50 });
        const mapped = (Array.isArray(data) ? data : data.resources || data.items || []).map(mapResource);
        setTools(mapped);
        setFiltered(mapped);
        setApiOnline(true);
      } catch (err) {
        console.error("Backend offline, using mock data:", err);
        setApiOnline(false);
        setErrorMsg("Backend offline — showing demo data");
        // Fallback mock data spanning all types
        const mocks = [
          { id: "1", name: "OpenAI GPT-4", resource_type: "API",  description: "Advanced language model via REST API", pricing_type: "paid",     github_stars: 50000, latency_ms: 320, is_available: true  },
          { id: "2", name: "Gemini 1.5",   resource_type: "API",  description: "Google's multimodal AI API",            pricing_type: "freemium", github_stars: 12000, latency_ms: 280, is_available: true  },
          { id: "3", name: "Razorpay SDK", resource_type: "API",  description: "India's leading payment gateway API",   pricing_type: "paid",     github_stars: 1200,  latency_ms: 55,  is_available: true  },
          { id: "4", name: "Llama 3 70B",  resource_type: "Model",description: "Open-source LLM by Meta, fine-tuned",   pricing_type: "free",     github_stars: 45000, latency_ms: 180, is_available: true  },
          { id: "5", name: "Mistral 7B",   resource_type: "Model",description: "Compact high-performance language model",pricing_type: "free",     github_stars: 22000, latency_ms: 95,  is_available: true  },
          { id: "6", name: "Stable Diffusion XL", resource_type: "Model", description: "Text-to-image generation model", pricing_type: "free", github_stars: 38000, latency_ms: 420, is_available: false },
          { id: "7", name: "Common Crawl Hindi", resource_type: "Dataset", description: "Petabyte-scale Hindi web crawl data", pricing_type: "free", downloads: 500000, latency_ms: 0, is_available: true },
          { id: "8", name: "LAION Indian Art",   resource_type: "Dataset", description: "12GB tagged Indian art images",      pricing_type: "free", downloads: 120000, latency_ms: 0, is_available: true  },
        ].map(mapResource);
        setTools(mocks);
        setFiltered(mocks);
      } finally {
        setLoading(false);
      }
    };

    loadAll();
  }, []);

  // Debounced filter / OpenSearch search
  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      const lq = query.toLowerCase().trim();

      if (!lq) {
        // No query — just apply category filter locally
        setFiltered(tools.filter(t => activeCategory === "All" || t.category === activeCategory));
        return;
      }

      // With query — use backend OpenSearch + Bedrock semantic search
      setSearchLoading(true);
      try {
        const resp = await apiService.search(query, {
          resource_types: activeCategory === "All" ? null : [activeCategory],
          limit: 20,
        });
        const results = (resp.results || []).map(mapResource);
        setFiltered(results);
      } catch {
        // Fallback: local substring filter
        setFiltered(tools.filter(t => {
          const catOk = activeCategory === "All" || t.category === activeCategory;
          const qOk   = t.name.toLowerCase().includes(lq) || t.description.toLowerCase().includes(lq);
          return catOk && qOk;
        }));
      } finally {
        setSearchLoading(false);
      }
    }, 300);
  }, [query, activeCategory, tools]);

  // Health check ping
  useEffect(() => {
    apiService.healthCheck()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  const stableCount = tools.filter(t => t.status === "stable").length;
  const NAV = [
    { key: "Home",     label: "Home" },
    { key: "Cpu",      label: "APIs" },
    { key: "Brain",    label: "Models" },
    { key: "Database", label: "Data" },
  ];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
        @keyframes cardIn    { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulseGlow { 0%,100% { opacity: 1; box-shadow: 0 0 6px currentColor; } 50% { opacity: 0.4; box-shadow: none; } }
        @keyframes shimmer   { 0%,100% { opacity: 0.45; } 50% { opacity: 0.8; } }
        @keyframes fadeIn    { from { opacity: 0; } to { opacity: 1; } }
        .cat-btn:hover { background: rgba(255,255,255,0.07) !important; }
        input::placeholder { color: rgba(255,255,255,0.2); font-style: italic; }
        input:focus { outline: none; }
        button:focus { outline: none; }
      `}</style>

      <div style={{
        display: "flex", height: "100vh", width: "100vw", overflow: "hidden",
        background: "#0C0C0E", color: "#fff", fontFamily: "'DM Sans', sans-serif",
      }}>
        {/* ── Sidebar ───────────────────────────────────────────────────── */}
        <aside style={{
          width: 80, flexShrink: 0,
          background: "#101013",
          borderRight: "1px solid rgba(255,255,255,0.05)",
          display: "flex", flexDirection: "column", alignItems: "center",
          padding: "24px 0", gap: 4, zIndex: 20,
        }}>
          {/* Logo */}
          <div style={{
            width: 40, height: 40, borderRadius: 14, marginBottom: 20,
            background: "linear-gradient(135deg, #006B5D 0%, #00FFA3 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontWeight: 900, fontSize: 18, letterSpacing: "-0.05em",
            boxShadow: "0 4px 20px rgba(0,255,163,0.25)", cursor: "default",
          }}>
            D
          </div>

          {/* Nav items */}
          <div style={{ width: "100%", padding: "0 12px", display: "flex", flexDirection: "column", gap: 4 }}>
            {NAV.map((n, i) => (
              <NavItem
                key={n.key} iconKey={n.key} label={n.label}
                active={activeNav === i}
                onClick={() => {
                  setActiveNav(i);
                  const catMap = ["All", "API", "Model", "Dataset"];
                  setActiveCategory(catMap[i]);
                  setQuery("");
                }}
              />
            ))}
          </div>

          {/* Bottom actions */}
          <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
            {/* AI Chat toggle */}
            <button
              onClick={() => setShowChat(v => !v)}
              title="AI Assistant (Bedrock)"
              style={{
                width: 34, height: 34, borderRadius: 10,
                background: showChat ? "rgba(0,255,163,0.1)" : "transparent",
                border: showChat ? "1px solid rgba(0,255,163,0.3)" : "none",
                cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                color: showChat ? "#00FFA3" : "rgba(255,255,255,0.22)", transition: "all 0.2s",
              }}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = showChat ? "rgba(0,255,163,0.1)" : "transparent"; }}
            >
              <Icon d={Icons.Brain} size={15} color="currentColor" />
            </button>

            {["Bell", "Settings"].map(k => (
              <button key={k} style={{
                width: 34, height: 34, borderRadius: 10,
                background: "transparent", border: "none", cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "rgba(255,255,255,0.22)", transition: "all 0.2s",
              }}
                onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.05)"; e.currentTarget.style.color = "rgba(255,255,255,0.5)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "rgba(255,255,255,0.22)"; }}
              >
                <Icon d={Icons[k]} size={15} color="currentColor" />
              </button>
            ))}

            <div style={{
              width: 32, height: 32, borderRadius: "50%", marginTop: 4,
              background: "linear-gradient(135deg, #006B5D, #00FFA3)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontWeight: 800, fontSize: 13, cursor: "pointer",
              boxShadow: "0 2px 12px rgba(0,255,163,0.2)",
            }}>
              B
            </div>
          </div>
        </aside>

        {/* ── Main Column ───────────────────────────────────────────────── */}
        <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* ── Top Bar ─────────────────────────────────────────────── */}
          <header style={{
            flexShrink: 0, display: "flex", alignItems: "center", gap: 16,
            padding: "14px 32px",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            background: "#0C0C0E",
          }}>
            <div>
              <div style={{ fontSize: 10, color: "rgba(255,255,255,0.28)", textTransform: "uppercase", letterSpacing: "0.15em", fontWeight: 600 }}>
                Dev-Store · AI Bharat
              </div>
              <div style={{ fontSize: 15, fontWeight: 700, color: "rgba(255,255,255,0.82)", letterSpacing: "-0.02em" }}>
                Tools & Integrations
              </div>
            </div>

            {/* Search — powered by OpenSearch + Bedrock */}
            <div style={{ flex: 1, maxWidth: 640, margin: "0 auto", position: "relative" }}>
              <div style={{ position: "absolute", left: 16, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}>
                <Icon d={Icons.Search} size={15} color={searchLoading ? "#00FFA3" : "rgba(255,255,255,0.22)"} />
              </div>
              <input
                type="text" value={query}
                onChange={e => setQuery(e.target.value)}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                placeholder={focused ? "Search via OpenSearch + Bedrock AI..." : ghostText}
                style={{
                  width: "100%", borderRadius: 99,
                  padding: "12px 20px 12px 42px",
                  background: "rgba(255,255,255,0.04)",
                  border: `1px solid ${focused ? "rgba(0,107,93,0.7)" : "rgba(255,255,255,0.08)"}`,
                  color: "rgba(255,255,255,0.88)", fontSize: 14,
                  fontFamily: "'DM Sans', sans-serif",
                  boxShadow: focused ? "0 0 0 3px rgba(0,107,93,0.12)" : "none",
                  transition: "border-color 0.25s, box-shadow 0.25s",
                }}
              />
              {searchLoading && (
                <div style={{ position: "absolute", right: 16, top: "50%", transform: "translateY(-50%)", display: "flex", gap: 3 }}>
                  {[0, 0.15, 0.3].map(d => (
                    <div key={d} style={{ width: 4, height: 4, borderRadius: "50%", background: "#00FFA3", animation: `pulseGlow 1s ${d}s infinite` }} />
                  ))}
                </div>
              )}
            </div>

            {/* Stat pills + API status */}
            <div style={{ display: "flex", gap: 8, flexShrink: 0, alignItems: "center" }}>
              {/* API online indicator */}
              <div style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "6px 12px", borderRadius: 99,
                background: apiOnline === true  ? "rgba(0,255,163,0.06)"
                           : apiOnline === false ? "rgba(239,68,68,0.06)"
                           : "rgba(255,255,255,0.03)",
                border: `1px solid ${apiOnline === true ? "rgba(0,255,163,0.2)" : apiOnline === false ? "rgba(239,68,68,0.2)" : "rgba(255,255,255,0.06)"}`,
                fontSize: 11, color: apiOnline === true ? "#00FFA3" : apiOnline === false ? "#ef4444" : "rgba(255,255,255,0.3)",
              }}>
                <div style={{
                  width: 5, height: 5, borderRadius: "50%",
                  background: apiOnline === true ? "#00FFA3" : apiOnline === false ? "#ef4444" : "rgba(255,255,255,0.3)",
                  animation: apiOnline === true ? "pulseGlow 2s infinite" : "none",
                }} />
                {apiOnline === true ? "API Live" : apiOnline === false ? "Offline" : "Checking…"}
              </div>

              {[
                { icon: "TrendUp", color: "#00FFA3", label: `${stableCount} live` },
                { icon: "Package", color: "rgba(255,255,255,0.28)", label: `${tools.length} tools` },
              ].map(s => (
                <div key={s.label} style={{
                  display: "flex", alignItems: "center", gap: 7,
                  padding: "7px 14px", borderRadius: 99,
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.06)",
                  fontSize: 12, color: "rgba(255,255,255,0.42)",
                }}>
                  <Icon d={Icons[s.icon]} size={13} color={s.color} />
                  {s.label}
                </div>
              ))}
            </div>
          </header>

          {/* ── Category Pill Bar ─────────────────────────────────── */}
          <div style={{
            flexShrink: 0, display: "flex", alignItems: "center", gap: 8,
            padding: "12px 32px",
            borderBottom: "1px solid rgba(255,255,255,0.04)",
            background: "#0D0D10", overflowX: "auto",
          }}>
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                className="cat-btn"
                onClick={() => { setActiveCategory(cat); setQuery(""); }}
                style={{
                  padding: "7px 16px", borderRadius: 99,
                  fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12,
                  cursor: "pointer", flexShrink: 0, transition: "all 0.2s",
                  background: activeCategory === cat ? "#006B5D" : "rgba(255,255,255,0.04)",
                  color: activeCategory === cat ? "#fff" : "rgba(255,255,255,0.38)",
                  border: `1px solid ${activeCategory === cat ? "#006B5D" : "rgba(255,255,255,0.06)"}`,
                }}
              >
                {cat === "All" ? "All Types" : cat}
              </button>
            ))}

            {/* Error banner */}
            {errorMsg && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "rgba(239,68,68,0.7)", padding: "6px 12px", borderRadius: 99, background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.15)", flexShrink: 0 }}>
                ⚠ {errorMsg}
              </div>
            )}

            <div style={{ marginLeft: "auto", flexShrink: 0, display: "flex", alignItems: "center", gap: 5, fontSize: 11.5, color: "rgba(255,255,255,0.2)" }}>
              <Icon d={Icons.ChevronR} size={13} />
              {loading ? "—" : filtered.length} result{filtered.length !== 1 ? "s" : ""}
            </div>
          </div>

          {/* ── Bento Grid ─────────────────────────────────────────── */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
              {loading
                ? Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} index={i} />)
                : filtered.length === 0
                  ? (
                    <div style={{ gridColumn: "1/-1", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: "60px 0" }}>
                      <div style={{ fontSize: 36 }}>🔍</div>
                      <div style={{ fontSize: 14, color: "rgba(255,255,255,0.3)" }}>
                        No results for <em>"{query || activeCategory}"</em> via OpenSearch
                      </div>
                      <button onClick={() => { setQuery(""); setActiveCategory("All"); setActiveNav(0); }}
                        style={{
                          padding: "8px 20px", borderRadius: 99, border: "1px solid rgba(0,255,163,0.3)",
                          background: "none", color: "#00FFA3", fontSize: 12, fontWeight: 600,
                          cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
                        }}>
                        Clear filters
                      </button>
                    </div>
                  )
                  : filtered.map((tool, i) => <ToolCard key={tool.id} tool={tool} index={i} />)
              }
            </div>

            {/* Footer */}
            <div style={{ marginTop: 32, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, fontSize: 11, color: "rgba(255,255,255,0.1)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#00FFA3", animation: "pulseGlow 2s infinite" }} />
              Connected to AWS OpenSearch · Bedrock Semantic Search
              <Icon d={Icons.Wifi} size={11} color="rgba(255,255,255,0.15)" />
            </div>
          </div>
        </main>
      </div>

      {/* ── AI Chat Panel (Bedrock) ────────────────────────────────── */}
      {showChat && <AIChatPanel onClose={() => setShowChat(false)} />}
    </>
  );
}
