'use client';

// Dev-Store Dashboard — Glassmorphism + Neon Accents V2.0
import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import useSWR from "swr";
import apiService from "@/lib/api";
import useWindowSize from "@/components/hooks/useWindowSize";
import Tooltip from "@/components/Tooltip";
import "./DevStoreDashboard.css";

// ─── Global Constants ────────────────────────────────────────────────────────
const A = "#3B82F6"; // Primary Accent
const AL = "#60A5FA"; // Primary Accent Light
const CATEGORIES = ["All", "API", "Model", "Dataset"];
const Icon = ({ d, size = 16, color = "currentColor", strokeWidth = 1.75 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d)
      ? d.map((path, i) => <path key={i} d={path} />)
      : <path d={d} />}
  </svg>
);

const Icons = {
  Home: "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M9 22V12h6v10",
  Cpu: ["M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18", "M9 9h1v1H9zm4 0h1v1h-1zm4 0h1v1h-1zM9 13h1v1H9zm4 0h1v1h-1zm4 0h1v1h-1z"],
  Database: ["M12 2a9 3 0 1 0 0 6 9 3 0 0 0 0-6z", "M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5", "M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"],
  Layers: ["M12 2L2 7l10 5 10-5-10-5z", "M2 17l10 5 10-5", "M2 12l10 5 10-5"],
  Search: ["M21 21l-4.35-4.35", "M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0"],
  Bell: ["M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9", "M13.73 21a2 2 0 0 1-3.46 0"],
  Settings: ["M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z", "M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"],
  Star: "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
  Zap: "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
  Copy: ["M20 9H11a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2v-9a2 2 0 0 0-2-2z", "M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 0 2 2v1"],
  Check: "M20 6L9 17l-5-5",
  Activity: "M22 12h-4l-3 9L9 3l-3 9H2",
  TrendUp: "M23 6l-9.5 9.5-5-5L1 18 M17 6h6v6",
  Package: ["M16.5 9.4l-9-5.17", "M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z", "M3.27 6.96L12 12.01l8.73-5.05 M12 22.08V12"],
  ChevronR: "M9 18l6-6-6-6",
  Wifi: ["M5 12.55a11 11 0 0 1 14.08 0", "M1.42 9a16 16 0 0 1 21.16 0", "M8.53 16.11a6 6 0 0 1 6.95 0", "M12 20h.01"],
  Brain: ["M9.5 2a2.5 2.5 0 0 1 5 0", "M9.5 2C7.567 2 6 3.567 6 5.5v1C4.343 6.5 3 7.843 3 9.5S4.343 12.5 6 12.5v1c0 1.933 1.567 3.5 3.5 3.5s3.5-1.567 3.5-3.5v-1c1.657 0 3-1.343 3-3S16.657 6.5 15 6.5v-1C15 3.567 13.433 2 11.5 2H9.5z"],
  Globe: ["M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2z", "M2 12h20", "M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"],
  Send: "M22 2L11 13 M22 2L15 22l-4-9-9-4 22-7z",
  Sun: ["M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z", "M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"],
  Moon: "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z",
  Menu: ["M4 6h16", "M4 12h16", "M4 18h16"],
  X: ["M18 6L6 18", "M6 6l12 12"],
  External: ["M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6", "M15 3h6v6", "M10 14L21 3"],
  Github: "M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z",
  Download: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3",
  Plus: "M12 5v14M5 12h14",
  Trophy: "M6 9V2h12v7M6 13a3 3 0 0 1 6 0M6 13a3 3 0 0 0-6 0M18 13a3 3 0 0 1 6 0",
  Play: ["M5 3l14 9-14 9z"],
  LayoutGrid: ["M3 3h7v7H3z", "M14 3h7v7h-7z", "M14 14h7v7h-7z", "M3 14h7v7H3z"],
  Flask: ["M9 3h6v4l4.5 9.5a2 2 0 0 1-1.7 3.5H6.2a2 2 0 0 1-1.7-3.5L9 7V3z", "M9 3v4", "M15 3v4"]
};

// ─── Constants ────────────────────────────────────────────────────────────────
const GHOST_TEXTS = [
  "Bhai, best payment gateway batao...",
  "Looking for a ML Model API?",
  "Kaunsa dataset use karein?",
  "Find low-latency inference APIs...",
  "Best open-source model for NLP?",
  "Search with AI...",
];

// Accent colors & emojis per resource type
const TYPE_META = {
  API: { color: "#3B82F6", emoji: "🔌" },
  Model: { color: "#A855F7", emoji: "🧠" },
  Dataset: { color: "#F59E0B", emoji: "🗄️" },
};

const PRICING_META = {
  free: { color: "#00FFA3", label: "free" },
  paid: { color: "#3B82F6", label: "paid" },
  freemium: { color: "#F59E0B", label: "freemium" },
};

const MOCK_TOOLS = [
  { id: "1", name: "OpenAI GPT-4", resource_type: "API", description: "Advanced language model via REST API. Best for high-accuracy NLP tasks.", pricing_type: "paid", github_stars: 82000, downloads: 1200000, latency_ms: 320, is_available: true, rank: 1 },
  { id: "2", name: "Gemini 1.5 Pro", resource_type: "API", description: "Google's multimodal AI API with 1M+ token context window.", pricing_type: "freemium", github_stars: 15400, downloads: 450000, latency_ms: 280, is_available: true, rank: 2 },
  { id: "3", name: "Razorpay SDK Bharat", resource_type: "API", description: "India's leading payment gateway API, optimized for local banks.", pricing_type: "paid", github_stars: 4200, downloads: 850000, latency_ms: 45, is_available: true, rank: 3 },
  { id: "4", name: "Llama 3 70B", resource_type: "Model", description: "Meta's flagship open-source LLM, fine-tuned for reasoning.", pricing_type: "free", github_stars: 45000, downloads: 210000, latency_ms: 180, is_available: true, rank: 4 },
  { id: "5", name: "Mistral 7B v0.3", resource_type: "Model", description: "Compact high-performance language model for edge deployment.", pricing_type: "free", github_stars: 28000, downloads: 350000, latency_ms: 92, is_available: true, rank: 5 },
  { id: "6", name: "Stable Diffusion XL", resource_type: "Model", description: "State-of-the-art text-to-image generation model.", pricing_type: "free", github_stars: 52000, downloads: 980000, latency_ms: 650, is_available: true, rank: 6 },
  { id: "7", name: "Common Crawl Hindi", resource_type: "Dataset", description: "Petabyte-scale Hindi web crawl data for training Indian LLMs.", pricing_type: "free", github_stars: 1200, downloads: 54000, latency_ms: 0, is_available: true, rank: 7 },
  { id: "8", name: "LAION Indian Scenes", resource_type: "Dataset", description: "Large-scale tagged dataset of Indian geographical and cultural scenes.", pricing_type: "free", github_stars: 840, downloads: 22000, latency_ms: 0, is_available: true, rank: 8 },
];

// Map a backend resource to ToolCard-compatible shape
function mapResource(r, index = 0) {
  const meta = TYPE_META[r.resource_type] || { color: "#6B7280", emoji: "📦" };
  const stars = r.github_stars ?? r.downloads ?? 0;
  const latencyRaw = parseFloat(r.latency_ms ?? r.p99_latency ?? 0);
  const latency = latencyRaw > 0 ? latencyRaw : Math.floor(Math.random() * 80 + 10);

  let installCommand = r.install_command || r.endpoint_url || "";
  if (!installCommand) {
    const slug = r.name?.replace(/\s+/g, "-").toLowerCase() || "resource";
    if (r.resource_type === "API") installCommand = `curl -X POST https://api.devstore.ai/v1/${slug}`;
    else if (r.resource_type === "Model") installCommand = `pip install devstore && dv.load("${slug}")`;
    else installCommand = `load_dataset("${slug}", split="train")`;
  }

  return {
    id: r.id || String(Math.random()),
    name: r.name,
    category: r.resource_type || "API",
    description: r.description || "",
    installCommand,
    latency,
    iconEmoji: meta.emoji,
    status: r.status || (r.is_available !== false ? "stable" : "down"),
    stars: r.stars || r.github_stars || Math.floor(Math.random() * 5000),
    downloads: r.downloads || Math.floor(Math.random() * 100000),
    rank: r.rank || (typeof index === 'number' && index < 10 ? index + 1 : 0),
    accentColor: meta.color,
    docsUrl: r.docs_url || "https://docs.example.com",
  };
}

// ─── Ghost-text hook ──────────────────────────────────────────────────────────
function useGhostText(active) {
  const [text, setText] = useState("");
  const idx = useRef(0);
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
        phase.current === "typing" ? SPEEDS.typing :
          phase.current === "erasing" ? SPEEDS.erasing :
            phase.current === "pause" ? SPEEDS.pause : SPEEDS.gap;

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
function ToolCard({ tool, index, isDark = true }) {
  const [copied, setCopied] = useState(false);
  const [hovered, setHovered] = useState(false);
  const dk = isDark;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(tool.installCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { }
  };

  const fmt = (n) => n >= 1000000 ? (n / 1000000).toFixed(1) + "M" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
  const pricing = PRICING_META[tool.pricingType] || PRICING_META.free;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered
          ? dk ? `linear-gradient(145deg, ${tool.accentColor}15 0%, rgba(2,6,23,0.7) 100%)` : `linear-gradient(145deg, ${tool.accentColor}08 0%, rgba(255,255,255,0.7) 100%)`
          : dk ? "rgba(2,6,23,0.6)" : "rgba(255,255,255,0.6)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        border: `1px solid ${hovered ? tool.accentColor + "40" : (dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)")}`,
        borderRadius: 24,
        padding: "22px 22px 20px",
        display: "flex", flexDirection: "column", gap: 14,
        position: "relative", overflow: "hidden",
        height: "100%", minHeight: 240,
        transform: hovered ? "scale(1.02)" : "scale(1)",
        zIndex: hovered ? 40 : 1,
        boxShadow: hovered
          ? dk ? `0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px ${tool.accentColor}15, inset 0 1px 0 rgba(255,255,255,0.04)` : `0 20px 60px rgba(59,130,246,0.08), 0 0 0 1px ${tool.accentColor}10`
          : dk ? "0 2px 12px rgba(0,0,0,0.3)" : "0 2px 12px rgba(59,130,246,0.06)",
        transition: "all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
        animationDelay: `${index * 60}ms`,
        animation: "cardIn 0.4s ease both",
      }}
    >
      {/* Top shimmer line */}
      <div style={{
        position: "absolute", top: 0, left: "10%", right: "10%", height: 1,
        background: hovered
          ? `linear-gradient(90deg, transparent, ${tool.accentColor}60, transparent)`
          : dk ? `linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent)` : `linear-gradient(90deg, transparent, rgba(59,130,246,0.1), transparent)`,
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
            {tool.rank > 0 && tool.rank <= 10 && (
              <div style={{
                fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: 700,
                color: tool.accentColor || A, textTransform: "uppercase", letterSpacing: "0.05em",
                textShadow: `0 0 8px ${tool.accentColor}66`,
                marginBottom: 2
              }}>
                #{tool.rank} IN {tool.category}S
              </div>
            )}
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontWeight: 700, fontSize: 15, color: dk ? "rgba(255,255,255,0.92)" : "rgba(0,0,0,0.95)", letterSpacing: "-0.02em" }}>
              {tool.name}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 3 }}>
              <span style={{ fontSize: 11, color: dk ? "rgba(255,255,255,0.32)" : "rgba(0,0,0,0.65)", fontWeight: 500 }}>
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
        fontSize: 13, color: dk ? "rgba(255,255,255,0.42)" : "rgba(0,0,0,0.7)", lineHeight: 1.55,
        fontFamily: "'DM Sans', sans-serif",
        display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden",
      }}>
        {tool.description || "No description available."}
      </p>

      {/* Score bar (shown when from search results) */}
      {tool.score != null && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: dk ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.55)", fontFamily: "'Fira Code', monospace" }}>AI Score</span>
          <div style={{ flex: 1, height: 4, borderRadius: 99, background: dk ? "rgba(255,255,255,0.06)" : "rgba(59,130,246,0.08)", overflow: "hidden" }}>
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
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", margin: "auto 0 16px 0" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4, color: dk ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.65)", fontSize: 12, background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", padding: "4px 8px", borderRadius: 6, border: dk ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.1)" }}>
          <Icon d={Icons.Globe} size={12} />
          <span style={{ fontWeight: 600 }}>{tool.provider}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, color: dk ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.65)", fontSize: 12, background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", padding: "4px 8px", borderRadius: 6, border: dk ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.1)" }}>
          <Icon d={Icons.Activity} size={11} />
          <span style={{ fontFamily: "'Fira Code', monospace" }}>{tool.latency}ms (p99)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, color: tool.accentColor, fontSize: 12, background: `${tool.accentColor}08`, padding: "4px 8px", borderRadius: 6, border: `1px solid ${tool.accentColor}15` }}>
          <Icon d={Icons.Star} size={11} fill={tool.accentColor} />
          <span style={{ fontWeight: 600 }}>{fmt(tool.stars)}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, color: dk ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.5)", fontSize: 12, background: dk ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)", padding: "4px 8px", borderRadius: 6, border: dk ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(0,0,0,0.06)" }}>
          <Icon d={Icons.Download} size={11} />
          <span>{fmt(tool.downloads)}</span>
        </div>
      </div>

      {/* Install / Endpoint box with integrated Copy */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          flex: 1, position: "relative",
          background: dk ? "rgba(2,6,23,0.9)" : "rgba(241,245,249,0.9)",
          borderRadius: 12, padding: "10px 12px",
          border: `1px solid ${dk ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)"}`,
          overflow: "hidden",
        }}>
          <code style={{
            fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
            display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
            paddingRight: hovered ? 32 : 0, transition: "padding 0.2s",
          }}>
            {(() => {
              const cmd = tool.installCommand;
              const keywords = ["pip", "install", "curl", "wget", "load_dataset", "dv.load", "-X", "POST"];
              return cmd.split(" ").map((word, i) => {
                const isKey = keywords.some(k => word.includes(k));
                const isUrl = word.includes("http");
                let color = dk ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)";
                if (isKey) color = tool.accentColor;
                if (isUrl) color = dk ? "rgba(147,197,253,0.8)" : "rgba(37,99,235,0.8)";
                return <span key={i} style={{ color, marginRight: 4 }}>{word}</span>;
              });
            })()}
          </code>

          {hovered && (
            <button
              onClick={handleCopy}
              style={{
                position: "absolute", top: 6, right: 6, width: 28, height: 28, borderRadius: 8,
                background: dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)",
                border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer",
                color: copied ? "#00FFA3" : (dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)"),
                transition: "all 0.2s", backdropFilter: "blur(4px)", zIndex: 10
              }}>
              <Icon d={copied ? Icons.Check : Icons.Copy} size={13} />
            </button>
          )}
        </div>

        <a href={tool.docsUrl} target="_blank" rel="noopener noreferrer" style={{
          background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", borderRadius: 10, padding: "8px 12px",
          border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, cursor: "pointer",
          color: dk ? "#fff" : "#000", fontSize: 11, fontWeight: 700, fontFamily: "'DM Sans', sans-serif",
          textDecoration: "none"
        }}>
          Docs
        </a>
      </div>
    </div>
  );
}

// ─── Skeleton Card ────────────────────────────────────────────────────────────
function SkeletonCard({ index, isDark = true }) {
  const dk = isDark;
  return (
    <div style={{
      background: dk ? "rgba(2, 6, 23, 0.6)" : "rgba(255,255,255,0.6)", borderRadius: 24, padding: "22px 22px 20px",
      backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)",
      border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)"}`,
      minHeight: 240,
      animationDelay: `${index * 80}ms`, animation: "shimmer 1.6s ease infinite",
    }}>
      {[44, 10, 13, 10, 10, 40].map((h, i) => (
        <div key={i} style={{
          height: h, borderRadius: 8, background: dk ? "rgba(255,255,255,0.04)" : "rgba(59,130,246,0.06)",
          marginBottom: 14, width: i === 1 ? "60%" : i === 2 ? "85%" : i === 3 ? "40%" : "100%",
        }} />
      ))}
    </div>
  );
}

// ─── Workbench / Logic Canvas (Solution Blueprint) ───────────────────────────
function ResourceWorkbench({ tool, onClose, isDark }) {
  const dk = isDark;
  const [pulse, setPulse] = useState(false);
  const isModel = tool.category === "Model";

  // Simplified Blueprint View
  const Blueprint = () => (
    <div style={{ flex: 1, position: "relative", background: dk ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.02)", borderRadius: 24, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", border: `1px solid ${dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"}` }}>
      <div style={{ position: "absolute", inset: 0, opacity: 0.1, backgroundImage: `radial-gradient(${dk ? "#fff" : "#000"} 1px, transparent 1px)`, backgroundSize: "30px 30px" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 60, zIndex: 1, padding: 40 }}>
        {/* Dataset Node */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
          <div style={{ width: 68, height: 68, borderRadius: 18, background: "rgba(245, 158, 11, 0.15)", border: "1px solid rgba(245, 158, 11, 0.3)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28 }}>🗄️</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }}>Dataset</div>
        </div>
        <Icon d={Icons.ChevronR} size={28} color={tool.accentColor} />
        {/* Model Node (Featured) */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, transform: "scale(1.2)" }}>
          <div style={{ width: 90, height: 90, borderRadius: 28, background: `${tool.accentColor}25`, border: `2px solid ${tool.accentColor}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36, boxShadow: `0 0 30px ${tool.accentColor}50`, position: "relative" }}>
            {tool.iconEmoji}
            <div style={{ position: "absolute", top: -5, right: -5, width: 24, height: 24, borderRadius: "50%", background: "#00FFA3", border: "3px solid #0a1020", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Icon d={Icons.Check} size={12} color="#000" strokeWidth={3} />
            </div>
          </div>
          <div style={{ fontSize: 15, fontWeight: 800, color: tool.accentColor, marginTop: 8 }}>{tool.name}</div>
        </div>
        <Icon d={Icons.ChevronR} size={28} color={tool.accentColor} />
        {/* API / App Node */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
          <div style={{ width: 68, height: 68, borderRadius: 18, background: "rgba(59, 130, 246, 0.15)", border: "1px solid rgba(59, 130, 246, 0.3)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28 }}>🔌</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }}>App/API</div>
        </div>
      </div>
      <div style={{ position: "absolute", bottom: 20, left: 24, padding: "10px 20px", borderRadius: 99, background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", border: `1px solid ${dk ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`, fontSize: 12, fontWeight: 700, color: dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", gap: 8 }}>
        <Icon d={Icons.LayoutGrid} size={14} /> Solution Blueprint: Active Flow
      </div>
    </div>
  );

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", background: dk ? "rgba(2, 6, 23, 0.9)" : "rgba(240, 244, 255, 0.9)", backdropFilter: "blur(32px)", animation: "fadeIn 0.3s ease-out" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Header Bar */}
        <div style={{ padding: "16px 40px", background: dk ? "rgba(10,16,32,0.8)" : "rgba(255,255,255,0.8)", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ width: 50, height: 50, borderRadius: 16, background: `${tool.accentColor}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>{tool.iconEmoji}</div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.03em" }}>{tool.name} <span style={{ opacity: 0.3, fontWeight: 500, marginLeft: 8 }}>Model Workbench</span></div>
              <div style={{ fontSize: 13, opacity: 0.5 }}>{tool.provider} • Verified Framework</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 16 }}>
            <button style={{ padding: "10px 24px", borderRadius: 12, background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", border: "none", fontWeight: 700, cursor: "pointer", fontSize: 13 }}>Share Blueprint</button>
            <button onClick={onClose} style={{ width: 44, height: 44, borderRadius: 12, border: "none", background: "rgba(239, 68, 68, 0.1)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "#ef4444" }}><Icon d={Icons.X} size={22} /></button>
          </div>
        </div>

        {/* Workspace */}
        <div style={{ flex: 1, display: "flex" }}>
          {/* Left: Logic Canvas */}
          <div style={{ flex: 1.2, padding: 40, borderRight: `1px solid ${dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"}`, display: "flex", flexDirection: "column", gap: 32 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
              <div>
                <h3 style={{ fontSize: 16, fontWeight: 800, marginBottom: 4 }}>Architectural Flow</h3>
                <p style={{ fontSize: 13, opacity: 0.5 }}>Connect datasets and output endpoints.</p>
              </div>
              <div style={{ display: "flex", gap: 8 }}>{["Datasets", "Transforms", "Apps"].map(t => <button key={t} style={{ padding: "6px 14px", borderRadius: 8, background: dk ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)", border: "none", fontSize: 11, fontWeight: 700, cursor: "pointer", opacity: 0.6 }}>+ {t}</button>)}</div>
            </div>
            <Blueprint />
            <div style={{ height: 180, background: dk ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)", borderRadius: 24, padding: 24, border: `1px solid ${dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"}` }}>
              <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 12, textTransform: "uppercase", color: tool.accentColor }}>Model Insights</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 24 }}>
                <div><div style={{ opacity: 0.4, fontSize: 11 }}>Precision</div><div style={{ fontWeight: 800, fontSize: 15 }}>FP16 / BF16</div></div>
                <div><div style={{ opacity: 0.4, fontSize: 11 }}>Context Window</div><div style={{ fontWeight: 800, fontSize: 15 }}>128k Tokens</div></div>
                <div><div style={{ opacity: 0.4, fontSize: 11 }}>Optimization</div><div style={{ fontWeight: 800, fontSize: 15 }}>Triton Kernel</div></div>
              </div>
            </div>
          </div>

          {/* Right: Technical Playground */}
          <div style={{ flex: 1, padding: 40, background: dk ? "rgba(0,0,0,0.15)" : "rgba(0,0,0,0.02)", display: "flex", flexDirection: "column", gap: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#00FFA3", boxShadow: "0 0 10px #00FFA3" }} />
              <h3 style={{ fontSize: 16, fontWeight: 800 }}>Inference Playground</h3>
            </div>

            <div style={{ flex: 1, background: dk ? "#020617" : "#fff", borderRadius: 24, border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, overflow: "hidden", display: "flex", flexDirection: "column", boxShadow: "0 10px 30px rgba(0,0,0,0.2)" }}>
              <div style={{ padding: "12px 20px", background: dk ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"}`, display: "flex", gap: 16 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: tool.accentColor }}>PROMPT</span>
                <span style={{ fontSize: 11, fontWeight: 700, opacity: 0.3 }}>RESPONSE</span>
              </div>
              <textarea
                placeholder="Type your prompt here..."
                style={{ flex: 1, background: "none", border: "none", padding: 24, paddingBottom: 12, resize: "none", fontSize: 14, fontFamily: "'JetBrains Mono', monospace", color: "inherit", outline: "none" }}
              />
              <div style={{ padding: 20, display: "flex", alignItems: "center", gap: 16 }}>
                <div style={{ flex: 1, height: 1, background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)" }} />
                <button
                  onMouseEnter={() => setPulse(true)}
                  onMouseLeave={() => setPulse(false)}
                  style={{
                    padding: "0 32px", height: 50, borderRadius: 14, background: tool.accentColor, border: "none", color: "#fff", fontWeight: 800, cursor: "pointer",
                    boxShadow: pulse ? `0 0 30px ${tool.accentColor}80` : "none", transition: "all 0.3s",
                    display: "flex", alignItems: "center", gap: 10, fontSize: 15
                  }}
                >
                  <Icon d={Icons.Play} size={18} fill="#fff" /> EXECUTE
                </button>
              </div>
            </div>

            <div style={{ height: 280, background: dk ? "#0a1020" : "#fff", borderRadius: 24, padding: 24, border: `1px solid ${dk ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`, overflowY: "auto", fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
              <div style={{ opacity: 0.3, marginBottom: 12, fontSize: 11, fontWeight: 800 }}>LIVE OUTPUT</div>
              <div style={{ color: "#00FFA3" }}>[SYSTEM]: Inializing model {tool.name}...</div>
              <div style={{ marginTop: 8 }}>Ready for input. Prompt context active.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


// ─── Nav Item ─────────────────────────────────────────────────────────────────
function NavItem({ iconKey, label, active, onClick, isDark = true }) {
  const [hov, setHov] = useState(false);
  const dk = isDark;
  const ac = "#3B82F6"; const al = "#60A5FA";
  const needsPulse = ["APIs", "Models", "Data"].includes(label);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        width: "100%", background: active ? `${ac}22` : hov ? (dk ? "rgba(255,255,255,0.04)" : "rgba(59,130,246,0.06)") : "transparent",
        border: "none", borderRadius: 16, padding: "12px 8px", cursor: "pointer",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
        position: "relative", transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
      }}
      className={active && needsPulse ? "neon-active" : ""}
    >
      {active && (
        <div style={{
          position: "absolute", left: -12, top: "50%", transform: "translateY(-50%)",
          width: 3, height: 28, background: ac, borderRadius: "0 3px 3px 0",
          boxShadow: `0 0 10px ${ac}`
        }} />
      )}
      <Icon
        d={Icons[iconKey]} size={18}
        color={active ? al : hov ? (dk ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.8)") : (dk ? "rgba(255,255,255,0.28)" : "rgba(0,0,0,0.6)")}
      />
      <span style={{
        fontSize: 9.5, fontWeight: 700, letterSpacing: "0.08em",
        color: active ? al : (dk ? "rgba(255,255,255,0.28)" : "rgba(0,0,0,0.7)"),
        textTransform: "uppercase", fontFamily: "'DM Sans', sans-serif",
      }}>
        {label}
      </span>
    </button>
  );
}

// ─── Settings Modal ───────────────────────────────────────────────────────────
function SettingsModal({ isDark, onClose, isHinglish, setIsHinglish, toggleTheme, notifySystems, setNotifySystems }) {
  const dk = isDark;
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 2000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.4)", backdropFilter: "blur(8px)", animation: "fadeIn 0.2s ease" }} onClick={onClose}>
      <div style={{ width: 400, background: dk ? "rgba(10, 16, 30, 0.85)" : "rgba(255,255,255,0.85)", backdropFilter: "blur(20px)", borderRadius: 24, border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, padding: 32, boxShadow: "0 40px 100px rgba(0,0,0,0.5)" }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 24 }}>
          <h2 style={{ fontSize: 18, fontWeight: 800 }}>System Settings</h2>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "inherit" }}><Icon d={Icons.X} size={20} /></button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Hinglish Mode</div>
              <div style={{ fontSize: 12, opacity: 0.5 }}>Semantic translation for docs</div>
            </div>
            <button onClick={() => setIsHinglish(!isHinglish)} style={{ width: 44, height: 24, borderRadius: 12, background: isHinglish ? "#3B82F6" : "rgba(128,128,128,0.2)", border: "none", cursor: "pointer", position: "relative", transition: "all 0.3s" }}>
              <div style={{ position: "absolute", top: 3, left: isHinglish ? 23 : 3, width: 18, height: 18, borderRadius: "50%", background: "#fff", transition: "all 0.3s" }} />
            </button>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Appearance</div>
              <div style={{ fontSize: 12, opacity: 0.5 }}>{dk ? "Dark" : "Light"} mode active</div>
            </div>
            <button onClick={toggleTheme} style={{ width: 44, height: 24, borderRadius: 12, background: dk ? "#3B82F6" : "#A855F7", border: "none", cursor: "pointer", position: "relative", transition: "all 0.3s" }}>
              <Icon d={dk ? Icons.Moon : Icons.Sun} size={12} color="white" style={{ position: "absolute", top: 6, left: dk ? 26 : 6 }} />
              <div style={{ position: "absolute", top: 3, left: dk ? 3 : 23, width: 18, height: 18, borderRadius: "50%", background: "#fff", transition: "all 0.3s" }} />
            </button>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontWeight: 700 }}>System Notifications</div>
              <div style={{ fontSize: 12, opacity: 0.5 }}>Health & API status alerts</div>
            </div>
            <button onClick={() => setNotifySystems(!notifySystems)} style={{ width: 44, height: 24, borderRadius: 12, background: notifySystems ? "#10B981" : "rgba(128,128,128,0.2)", border: "none", cursor: "pointer", position: "relative", transition: "all 0.3s" }}>
              <div style={{ position: "absolute", top: 3, left: notifySystems ? 23 : 3, width: 18, height: 18, borderRadius: "50%", background: "#fff", transition: "all 0.3s" }} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Auth Portal ──────────────────────────────────────────────────────────────
function AuthPortal({ isDark, onClose, onLogin }) {
  const dk = isDark;

  // Handle Escape key
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  return (
    <div
      style={{ position: "fixed", inset: 0, zIndex: 2000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.4)", backdropFilter: "blur(8px)", animation: "fadeIn 0.2s ease" }}
      onClick={onClose}
    >
      <div
        style={{ position: "relative", width: 440, background: dk ? "rgba(10, 16, 30, 0.85)" : "rgba(255,255,255,0.85)", backdropFilter: "blur(20px)", borderRadius: 32, border: `1px solid ${dk ? "rgba(255,255,255,0.12)" : "rgba(59,130,246,0.15)"}`, padding: 48, textAlign: "center", boxShadow: "0 40px 100px rgba(0,0,0,0.5)" }}
        onClick={e => e.stopPropagation()}
      >
        {/* 'X' Close Button */}
        <button
          onClick={onClose}
          style={{ position: "absolute", top: 24, right: 24, background: "none", border: "none", cursor: "pointer", color: dk ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.4)", display: "flex", alignItems: "center", justifyContent: "center", transition: "color 0.2s" }}
          onMouseEnter={e => e.currentTarget.style.color = dk ? "#fff" : "#000"}
          onMouseLeave={e => e.currentTarget.style.color = dk ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.4)"}
        >
          <Icon d={Icons.X} size={20} />
        </button>

        <div style={{ width: 70, height: 70, background: "rgba(59,130,246,0.1)", borderRadius: 20, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 24px" }}>
          <img src="/logo.png" style={{ width: "70%" }} alt="B" />
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 8 }}>Welcome to DevStore</h1>
        <p style={{ fontSize: 14, opacity: 0.5, marginBottom: 32 }}>Build, deploy, and scale with the world's most powerful Developer's marketplace.</p>

        <button
          onClick={() => onLogin({ name: "Bharat Dev", avatar: "https://i.pravatar.cc/150?u=bharat" })}
          style={{ width: "100%", height: 50, background: dk ? "#fff" : "#1a1a1a", color: dk ? "#000" : "#fff", borderRadius: 14, border: "none", fontWeight: 700, fontSize: 15, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 12, marginBottom: 16, transition: "transform 0.2s" }}
          onMouseEnter={e => e.currentTarget.style.transform = "translateY(-2px)"}
          onMouseLeave={e => e.currentTarget.style.transform = "translateY(0)"}>
          <Icon d={Icons.Github} size={20} /> Continue with GitHub
        </button>

        <div style={{ fontSize: 13, opacity: 0.4 }}>
          By continuing, you agree to our <a href="#" style={{ color: "inherit", textDecoration: "underline" }}>Terms</a> and <a href="#" style={{ color: "inherit", textDecoration: "underline" }}>Privacy Policy</a>.
        </div>
        <button onClick={onClose} style={{ marginTop: 32, background: "none", border: "none", cursor: "pointer", fontSize: 13, opacity: 0.6, textDecoration: "underline" }}>Skip for now</button>
      </div>
    </div>
  );
}

// ─── Resource Submission Modal ────────────────────────────────────────────────
function ResourceSubmissionModal({ isDark, onClose }) {
  const dk = isDark;
  const [step, setStep] = useState(1); // 1: Github, 2: Pre-fill, 3: Success
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(false);
  const [metadata, setMetadata] = useState(null);

  const handleSync = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setStep(2);
      // Mocked AI parsing result
      setMetadata({
        title: "Virtual Court AI",
        desc: "Legal model optimized for Bharat's court proceedings, handling both English and Hinglish transcription with high accuracy.",
        type: "Model",
        snippet: "model = dv.load('virtual-court-ai')\nresult = model.transcribe(audio_file)"
      });
    }, 1500);
  };

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 3000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.5)", backdropFilter: "blur(12px)", animation: "fadeIn 0.2s ease" }}>
      <div style={{ width: 500, background: dk ? "#0a1020" : "#fff", borderRadius: 32, border: `1px solid ${dk ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)"}`, padding: 40, boxShadow: "0 40px 100px rgba(0,0,0,0.5)", position: "relative" }}>
        <button onClick={onClose} style={{ position: "absolute", top: 24, right: 24, background: "none", border: "none", cursor: "pointer", color: "inherit", opacity: 0.4 }}><Icon d={Icons.X} size={20} /></button>

        {step === 1 && (
          <>
            <div style={{ width: 60, height: 60, background: "rgba(59,130,246,0.1)", borderRadius: 16, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 24 }}>
              <Icon d={Icons.Github} size={30} color={A} />
            </div>
            <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Submit to DevStore</h2>
            <p style={{ fontSize: 14, opacity: 0.6, marginBottom: 32 }}>Sync your repository and our AI will automatically prepare your listing.</p>

            <input
              placeholder="github.com/username/repo-name"
              value={repo}
              onChange={e => setRepo(e.target.value)}
              style={{ width: "100%", padding: "14px 20px", borderRadius: 14, background: dk ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)", border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, color: "inherit", marginBottom: 20, fontSize: 14 }}
            />

            <button
              onClick={handleSync}
              disabled={!repo || loading}
              style={{ width: "100%", height: 50, background: A, color: "#fff", borderRadius: 14, border: "none", fontWeight: 800, fontSize: 15, cursor: "pointer", opacity: (!repo || loading) ? 0.5 : 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}
            >
              {loading ? "Syncing..." : "Sync & Pre-fill Metadata"}
            </button>
          </>
        )}

        {step === 2 && (
          <>
            <div style={{ fontSize: 11, fontWeight: 800, color: A, textTransform: "uppercase", marginBottom: 8 }}>AI Parse Success</div>
            <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 20 }}>Review Listing</h2>

            <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 32 }}>
              <div>
                <label style={{ fontSize: 11, fontWeight: 700, opacity: 0.4 }}>RESOURCE TITLE</label>
                <input value={metadata.title} style={{ width: "100%", background: "none", border: "none", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, padding: "8px 0", fontSize: 16, fontWeight: 700, color: "inherit" }} />
              </div>
              <div>
                <label style={{ fontSize: 11, fontWeight: 700, opacity: 0.4 }}>DESCRIPTION (HINGLISH)</label>
                <textarea value={metadata.desc} style={{ width: "100%", background: "none", border: "none", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, padding: "8px 0", fontSize: 13, height: 60, resize: "none", color: "inherit" }} />
              </div>
              <div style={{ background: dk ? "#020617" : "#f1f5f9", padding: 16, borderRadius: 12 }}>
                <label style={{ fontSize: 10, fontWeight: 800, opacity: 0.3, display: "block", marginBottom: 8 }}>CODE SNIPPET</label>
                <code style={{ fontSize: 11, fontFamily: "monospace", opacity: 0.8, whiteSpace: "pre" }}>{metadata.snippet}</code>
              </div>
            </div>

            <button onClick={() => setStep(3)} style={{ width: "100%", height: 50, background: "#10B981", color: "#fff", borderRadius: 14, border: "none", fontWeight: 800, fontSize: 15, cursor: "pointer" }}>Submit Listing</button>
          </>
        )}

        {step === 3 && (
          <div style={{ textAlign: "center", padding: "20px 0" }}>
            <div style={{ width: 80, height: 80, background: "#10B98115", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 24px" }}>
              <Icon d={Icons.Check} size={40} color="#10B981" />
            </div>
            <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 12 }}>Resource Submitted!</h2>
            <p style={{ fontSize: 14, opacity: 0.6, marginBottom: 32 }}>Your application has been received and will be reviewed shortly by our moderators.</p>
            <button onClick={onClose} style={{ width: "100%", height: 50, background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", borderRadius: 14, border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, fontWeight: 700, cursor: "pointer", color: "inherit" }}>Back to Dashboard</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── AI Chat Panel ────────────────────────────────────────────────────────────
function AIChatPanel({ onClose, isDark = true, isMobile = false }) {
  const dk = isDark;
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
          results: results.map((r, idx) => mapResource(r, idx)),
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
      width: "100%", height: "100%",
      display: "flex", flexDirection: "column",
      fontSize: isMobile ? 14 : 13,
    }}>
      {/* Header */}
      <div style={{ padding: "20px", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`, display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg, #A855F7, #3B82F6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Icon d={Icons.Brain} size={18} color="white" />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: dk ? "#fff" : "#000" }}>Intent Discovery</div>
          <div style={{ fontSize: 10, color: dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#3B82F6", animation: "pulseGlow 2s infinite" }} />
            RAG Flow Active
          </div>
        </div>
        <button onClick={onClose} style={{ marginLeft: "auto", background: dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", border: "none", borderRadius: 8, width: 28, height: 28, color: dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)", cursor: "pointer" }}>×</button>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: m.type === "user" ? "flex-end" : "flex-start" }}>
            <div style={{
              maxWidth: "85%", padding: "10px 14px", borderRadius: m.type === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              background: m.type === "user" ? "rgba(59,130,246,0.25)" : (dk ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"),
              border: `1px solid ${m.type === "user" ? "rgba(59,130,246,0.3)" : (dk ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)")}`,
              fontSize: 13, color: dk ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.8)", lineHeight: 1.55,
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
            flex: 1, background: dk ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)",
            border: `1px solid ${dk ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
            borderRadius: 99, padding: "10px 16px", color: dk ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.8)", fontSize: 13,
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
  const [tools, setTools] = useState([]);
  const [filtered, setFiltered] = useState([]);
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeNav, setActiveNav] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [focused, setFocused] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [apiOnline, setApiOnline] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isHinglish, setIsHinglish] = useState(false);
  const [isTopChart, setIsTopChart] = useState(false);
  const [selectedTool, setSelectedTool] = useState(null);
  const [activeDiscovery, setActiveDiscovery] = useState("Trending"); // Trending, Top Free, Top Paid, Most Popular
  const [showSubmission, setShowSubmission] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);
  const [notifySystems, setNotifySystems] = useState(true);
  const debounceRef = useRef(null);
  const ghostText = useGhostText(!focused && !query);

  // Theme (default: dark)
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      const s = window.localStorage.getItem("ds-theme");
      return s ? s === "dark" : true;
    }
    return true;
  });
  const toggleTheme = () => setIsDark(p => {
    const next = !p;
    localStorage.setItem("ds-theme", next ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
    return next;
  });
  const dk = isDark;

  // Sync data-theme on mount
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
  }, []);

  // Responsive
  const { width } = useWindowSize();
  const isMobile = width < 768;
  const gridCols = width >= 1200 ? 4 : width >= 1024 ? 3 : width >= 768 ? 2 : 1;
  const pad = isMobile ? 16 : 32;

  // SWR-based trending fetcher
  const { data: trendingData, error: trendingError, isValidating: trendingLoading } = useSWR(
    !query ? [`trending`, activeCategory, isTopChart, activeDiscovery] : null,
    () => {
      const filters = { resource_type: activeCategory === "All" ? null : activeCategory, limit: 40 };
      if (activeDiscovery === "Top Free") {
        filters.pricing_type = "free";
        filters.sort = "popularity";
      } else if (activeDiscovery === "Top Paid") {
        filters.pricing_type = "paid";
        filters.sort = "popularity";
      } else if (activeDiscovery === "Most Popular") {
        filters.sort = "downloads";
      } else {
        filters.sort = "rank_score"; // Unified Ranking Algorithm
      }
      return apiService.getTrending(filters);
    },
    { revalidateOnFocus: false }
  );

  useEffect(() => {
    if (trendingLoading) {
      setLoading(true);
    } else if (!query && trendingData) {
      let results = (trendingData.results || []).map((r, idx) => mapResource(r, idx));
      if (isTopChart) {
        results = results.sort((a, b) => (a.rank || 999) - (b.rank || 999));
      }
      setFiltered(results);
      setLoading(false);
    } else if (!query && trendingError) {
      setApiOnline(false); setErrorMsg("Backend offline — showing demo data");
      const mappedMocks = MOCK_TOOLS.map((r, idx) => mapResource(r, idx));
      let sortedMocks = [...mappedMocks];

      if (activeDiscovery === "Trending") {
        sortedMocks.sort((a, b) => (b.stars + b.downloads) - (a.stars + a.downloads));
      } else if (activeDiscovery === "Top Free") {
        sortedMocks = sortedMocks.filter(r => r.pricingType === "free");
        sortedMocks.sort((a, b) => b.stars - a.stars);
      } else if (activeDiscovery === "Top Paid") {
        sortedMocks = sortedMocks.filter(r => r.pricingType === "paid");
        sortedMocks.sort((a, b) => b.stars - a.stars);
      } else if (activeDiscovery === "Most Popular") {
        sortedMocks.sort((a, b) => b.downloads - a.downloads);
      }

      if (isTopChart) sortedMocks.sort((a, b) => a.rank - b.rank);
      setFiltered(sortedMocks);
      setLoading(false);
    }
  }, [trendingData, trendingError, trendingLoading, query, activeCategory, isTopChart, activeDiscovery]);

  // Initial connectivity check
  useEffect(() => {
    apiService.healthCheck().then(() => setApiOnline(true)).catch(() => setApiOnline(false));
  }, []);

  // Search Debounce logic
  useEffect(() => {
    clearTimeout(debounceRef.current);
    const lq = query.toLowerCase().trim();
    if (!lq) {
      // If query is cleared, trending data handles the view
      setSearchLoading(false); // Ensure search loading is off when not searching
      return;
    }

    setSearchLoading(true);
    debounceRef.current = setTimeout(async () => {
      try {
        const resp = await apiService.search(query, { resource_types: activeCategory === "All" ? null : [activeCategory], limit: 40 });
        setFiltered((resp.results || []).map((r, idx) => mapResource(r, idx)));
        setApiOnline(true);
      } catch (err) {
        console.error("Search failed:", err);
        setApiOnline(false);
        setErrorMsg("Backend offline — showing local matches");

        // Use MOCK_TOOLS as source if backend is down and 'tools' is empty
        const sourceData = tools.length > 0 ? tools : MOCK_TOOLS;
        const mappedSource = sourceData.map((r, idx) => mapResource(r, idx));

        setFiltered(mappedSource.filter(t =>
          (activeCategory === "All" || t.category === activeCategory) &&
          (t.name.toLowerCase().includes(lq) || t.description.toLowerCase().includes(lq))
        ));
      } finally { setSearchLoading(false); }
    }, 400); // Increased debounce to 400ms for better UX
  }, [query, activeCategory, tools]); // Added 'tools' to dependency array for local fallback


  const stableCount = tools.filter(t => t.status === "stable").length;
  const NAV = [
    { key: "Home", label: "Home", tip: "Browse all resources" },
    { key: "Cpu", label: "APIs", tip: "Browse APIs" },
    { key: "Brain", label: "Models", tip: "Browse ML Models" },
    { key: "Database", label: "Data", tip: "Browse Datasets" },
  ];

  // sidebarContent is now rendered inline with CSS classes

  return (
    <>
      <div className="ds-shell">

        {/* Mobile overlay */}
        {isMobile && sidebarOpen && <div className="ds-mobile-overlay" onClick={() => setSidebarOpen(false)} />}

        {/* Gradient Orbs */}
        <div className="ds-orb ds-orb--1" />
        <div className="ds-orb ds-orb--2" />
        <div className="ds-orb ds-orb--3" />

        {/* Sidebar */}
        {(!isMobile || sidebarOpen) && (
          <aside className={`ds-sidebar ${isMobile ? "ds-sidebar--mobile" : ""} ${!isMobile && sidebarCollapsed ? "ds-sidebar--collapsed" : ""}`}>
            {!isMobile && (
              <button className="ds-sidebar__toggle" onClick={() => setSidebarCollapsed(c => !c)} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
                <Icon d={Icons.ChevronR} size={14} />
              </button>
            )}
            <div className="ds-sidebar__header">
              <div className="ds-sidebar__logo"><img src="/logo.png" alt="DevStore" /></div>
              <span className="ds-sidebar__brand ds-sidebar__label">DevStore</span>
              {isMobile && <button onClick={() => setSidebarOpen(false)} style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", color: "var(--ds-text-secondary)" }} aria-label="Close menu"><Icon d={Icons.X} size={18} /></button>}
            </div>

            <nav className="ds-sidebar__nav">
              {NAV.map((n, i) => (
                <button key={n.key} className={`ds-sidebar__nav-item ${activeNav === i ? "ds-sidebar__nav-item--active" : ""}`}
                  onClick={() => { setActiveNav(i); setActiveCategory(["All", "API", "Model", "Dataset"][i]); setQuery(""); if (isMobile) setSidebarOpen(false); }}>
                  <span className="ds-sidebar__icon-wrap">
                    <Icon d={Icons[n.key]} size={24} />
                  </span>
                  <span className="ds-sidebar__label">{n.label}</span>
                </button>
              ))}
            </nav>

            <div className="ds-sidebar__divider" />
            <div className="ds-sidebar__section-title">Quick Access</div>
            <nav className="ds-sidebar__nav">
              <button className="ds-sidebar__nav-item" onClick={() => setShowChat(true)}>
                <span className="ds-sidebar__icon-wrap"><Icon d={Icons.Brain} size={24} color="var(--ds-primary)" /></span>
                <span className="ds-sidebar__label">AI Assistant</span>
              </button>
              <button className="ds-sidebar__nav-item" onClick={() => setIsTopChart(!isTopChart)} style={isTopChart ? { color: "var(--ds-primary)" } : {}}>
                <span className="ds-sidebar__icon-wrap"><Icon d={Icons.LayoutGrid} size={24} /></span>
                <span className="ds-sidebar__label">Top Chart</span>
              </button>
            </nav>

            <div className="ds-sidebar__bottom">
              <button className="ds-sidebar__bottom-item" onClick={() => setShowSettings(true)} aria-label="Settings">
                <span className="ds-sidebar__icon-wrap"><Icon d={Icons.Settings} size={24} /></span>
                <span className="ds-sidebar__label">Settings</span>
              </button>
              <button className="ds-sidebar__bottom-item" onClick={() => isLoggedIn ? null : setShowAuth(true)} aria-label="Login">
                {isLoggedIn
                  ? <><div className="ds-topbar__user-avatar" style={{ width: 28, height: 28 }}><img src={user?.avatar} alt="" /></div> <span className="ds-sidebar__label">{user?.name}</span></>
                  : <><span className="ds-sidebar__icon-wrap"><Icon d={Icons.Github} size={24} /></span> <span className="ds-sidebar__label">Login</span></>
                }
              </button>
            </div>
          </aside>
        )}

        <main className="ds-main">
          <header style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: isMobile ? 10 : 16, padding: `14px ${pad}px`, flexWrap: isMobile ? "wrap" : "nowrap", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.05)" : "rgba(59,130,246,0.1)"}`, background: dk ? "rgba(8,12,24,0.8)" : "rgba(240,244,255,0.8)", backdropFilter: "blur(12px)" }}>
            {isMobile && (
              <Tooltip text="Open menu" position="bottom">
                <button onClick={() => setSidebarOpen(true)} style={{ background: "none", border: "none", cursor: "pointer", color: dk ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, borderRadius: 10 }}>
                  <Icon d={Icons.Menu} size={20} />
                </button>
              </Tooltip>
            )}
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 10, color: dk ? "rgba(255,255,255,0.28)" : "rgba(0,0,0,0.65)", textTransform: "uppercase", letterSpacing: "0.15em", fontWeight: 600 }}>Dev-Store · AI Bharat</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: dk ? "rgba(255,255,255,0.82)" : "rgba(0,0,0,0.9)", letterSpacing: "-0.02em" }}>Tools & Integrations</div>
            </div>

            {/* Search */}
            <div style={{ flex: isMobile ? "1 1 100%" : 1, maxWidth: 640, margin: isMobile ? "8px auto 4px" : "0 auto", position: "relative", order: isMobile ? 10 : 0, width: isMobile ? "100%" : "auto" }}>
              <div style={{ position: "absolute", left: 16, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}>
                <Icon d={Icons.Search} size={15} color={searchLoading ? A : (dk ? "rgba(255,255,255,0.22)" : "rgba(0,0,0,0.5)")} />
              </div>
              <input type="text" value={query} onChange={e => setQuery(e.target.value)} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
                placeholder={focused ? "Search via OpenSearch + Bedrock AI..." : ghostText}
                style={{ width: "100%", borderRadius: 99, padding: isMobile ? "10px 20px 10px 42px" : "12px 20px 12px 42px", background: dk ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.75)", border: `1px solid ${focused ? `${A}99` : (dk ? "rgba(255,255,255,0.08)" : "rgba(59,130,246,0.15)")}`, color: dk ? "rgba(255,255,255,0.88)" : "rgba(0,0,0,0.95)", fontSize: 14, fontFamily: "'DM Sans', sans-serif", boxShadow: focused ? `0 0 0 3px ${A}20` : "none", transition: "border-color 0.25s, box-shadow 0.25s" }} />
              {searchLoading && <div style={{ position: "absolute", right: 16, top: "50%", transform: "translateY(-50%)", display: "flex", gap: 3 }}>{[0, 0.15, 0.3].map(d => <div key={d} style={{ width: 4, height: 4, borderRadius: "50%", background: A, animation: `pulseGlow 1s ${d}s infinite` }} />)}</div>}
            </div>

            {/* Stats + Theme toggle */}
            <div style={{ display: "flex", gap: 8, flexShrink: 0, alignItems: "center" }}>
              {!isMobile && <>
                <Tooltip text="Backend API status" position="bottom">
                  <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 99, background: apiOnline === true ? "rgba(0,200,130,0.06)" : apiOnline === false ? "rgba(239,68,68,0.06)" : (dk ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)"), border: `1px solid ${apiOnline === true ? "rgba(0,200,130,0.2)" : apiOnline === false ? "rgba(239,68,68,0.2)" : (dk ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)")}`, fontSize: 11, color: apiOnline === true ? "#059669" : apiOnline === false ? "#ef4444" : (dk ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.6)") }}>
                    <div style={{ width: 5, height: 5, borderRadius: "50%", background: apiOnline === true ? "#059669" : apiOnline === false ? "#ef4444" : (dk ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.4)"), animation: apiOnline === true ? "pulseGlow 2s infinite" : "none" }} />
                    {apiOnline === true ? "API Live" : apiOnline === false ? "Offline" : "Checking…"}
                  </div>
                </Tooltip>
                {[{ icon: "TrendUp", color: "#059669", label: `${stableCount} live`, tip: "Live resources" }, { icon: "Package", color: dk ? "rgba(255,255,255,0.28)" : "rgba(0,0,0,0.55)", label: `${tools.length} tools`, tip: "Total tools" }].map(s => (
                  <Tooltip key={s.label} text={s.tip} position="bottom">
                    <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "7px 14px", borderRadius: 99, background: dk ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.6)", border: `1px solid ${dk ? "rgba(255,255,255,0.06)" : "rgba(59,130,246,0.1)"}`, fontSize: 12, color: dk ? "rgba(255,255,255,0.42)" : "rgba(0,0,0,0.7)" }}>
                      <Icon d={Icons[s.icon]} size={13} color={s.color} />{s.label}
                    </div>
                  </Tooltip>
                ))}
              </>}
              <Tooltip text="Toggle Hinglish Mode" position="bottom">
                <button onClick={() => setIsHinglish(!isHinglish)} style={{
                  padding: "0 12px", height: 38, borderRadius: 12, display: "flex", alignItems: "center", gap: 6,
                  background: isHinglish ? "linear-gradient(135deg, #F59E0B22, #F59E0B0A)" : (dk ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)"),
                  border: `1px solid ${isHinglish ? "#F59E0B55" : (dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)")}`,
                  color: isHinglish ? "#F59E0B" : (dk ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)"),
                  fontWeight: 700, fontSize: 13, cursor: "pointer", transition: "all 0.3s ease"
                }}>
                  🇮🇳 {isHinglish ? "Hinglish: ON" : "Hinglish: OFF"}
                </button>
              </Tooltip>
              <Tooltip text={dk ? "Switch to Light mode" : "Switch to Dark mode"} position="bottom">
                <button onClick={toggleTheme} style={{ width: 38, height: 38, borderRadius: 12, background: dk ? `linear-gradient(135deg,${A}22,${A}0A)` : `linear-gradient(135deg,${A}15,${A}08)`, border: `1px solid ${dk ? `${A}30` : `${A}20`}`, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: dk ? AL : A, transition: "all 0.3s ease", boxShadow: `0 2px 8px rgba(59,130,246,0.15)` }}>
                  <Icon d={dk ? Icons.Sun : Icons.Moon} size={16} />
                </button>
              </Tooltip>
              <button
                onClick={() => setShowSubmission(true)}
                style={{
                  height: 38, padding: isMobile ? "0 14px" : "0 20px", borderRadius: 12,
                  background: `linear-gradient(135deg, ${A}, ${AL})`, color: "#fff",
                  display: "flex", alignItems: "center", gap: 8, fontWeight: 800, fontSize: 13,
                  cursor: "pointer", border: "none", transition: "transform 0.2s, box-shadow 0.2s",
                  boxShadow: `0 4px 15px ${A}30`
                }}
                onMouseEnter={e => e.currentTarget.style.transform = "translateY(-1px)"}
                onMouseLeave={e => e.currentTarget.style.transform = "translateY(0)"}
              >
                <Icon d={Icons.Plus} size={16} color="#fff" />
                {!isMobile && "Create App"}
              </button>
            </div>
          </header>

          {/* New Horizontal Row of Filter Pills */}
          <div style={{ flexShrink: 0, padding: `8px ${pad}px`, background: dk ? "rgba(8,12,24,0.4)" : "rgba(240,244,255,0.4)", borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)"}` }}>
            <div style={{ maxWidth: 1600, margin: "0 auto", display: "flex", gap: 10, overflowX: "auto" }} className="hide-scroll">
              {["Trending", "Top Free", "Top Paid", "Most Popular"].map(f => (
                <button
                  key={f}
                  onClick={() => setActiveDiscovery(f)}
                  style={{
                    padding: "6px 14px", borderRadius: 99, fontSize: 11, fontWeight: 700,
                    cursor: "pointer", border: "1px solid transparent", transition: "all 0.2s",
                    background: activeDiscovery === f ? dk ? "rgba(255,255,255,0.1)" : "rgba(59,130,246,0.1)" : "transparent",
                    color: activeDiscovery === f ? A : dk ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.5)",
                    borderColor: activeDiscovery === f ? `${A}40` : "transparent"
                  }}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {/* Category Pill Bar */}
          <div className="hide-scroll" style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: isMobile ? 12 : 8, padding: isMobile ? `14px ${pad}px` : `12px ${pad}px`, borderBottom: `1px solid ${dk ? "rgba(255,255,255,0.04)" : "rgba(59,130,246,0.08)"}`, background: dk ? "rgba(10,15,30,0.6)" : "rgba(234,239,255,0.6)", overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "0 auto", maxWidth: 1600, width: "100%" }}>
              {CATEGORIES.map(cat => (
                <Tooltip key={cat} text={`Filter: ${cat === "All" ? "All Types" : cat}`} position="bottom">
                  <button className="cat-btn" onClick={() => { setActiveCategory(cat); setQuery(""); }}
                    style={{ padding: "7px 16px", borderRadius: 99, fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12, cursor: "pointer", flexShrink: 0, transition: "all 0.2s", background: activeCategory === cat ? A : (dk ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.85)"), color: activeCategory === cat ? "#fff" : (dk ? "rgba(255,255,255,0.38)" : "rgba(0,0,0,0.75)"), border: `1px solid ${activeCategory === cat ? A : (dk ? "rgba(255,255,255,0.06)" : "rgba(59,130,246,0.2)")}` }}>
                    {cat === "All" ? "All Types" : cat}
                  </button>
                </Tooltip>
              ))}
              {errorMsg && <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "rgba(239,68,68,0.7)", padding: "6px 12px", borderRadius: 99, background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.15)", flexShrink: 0 }}>⚠ {errorMsg}</div>}
              <div style={{ marginLeft: "auto", flexShrink: 0, display: "flex", alignItems: "center", gap: 5, fontSize: 11.5, color: dk ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.6)" }}>
                <Icon d={Icons.ChevronR} size={13} />{loading ? "—" : filtered.length} result{filtered.length !== 1 ? "s" : ""}
              </div>
            </div>
          </div>

          {/* Bento Grid Container */}
          <div style={{ flex: 1, overflowY: "auto", overflowX: "hidden", padding: `24px ${pad}px` }}>
            <div style={{ maxWidth: 1600, margin: "0 auto", width: "100%" }}>

              {/* Intent Discovery Stacked on Mobile */}
              {isMobile && showChat && (
                <div style={{ marginBottom: 24, borderRadius: 24, overflow: "hidden", border: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`, background: dk ? "rgba(10,16,32,0.6)" : "rgba(255,255,255,0.6)", backdropFilter: "blur(20px)" }}>
                  <AIChatPanel onClose={() => setShowChat(false)} isDark={dk} isMobile={isMobile} />
                </div>
              )}

              <div style={{
                display: "grid",
                gridTemplateColumns: width >= 1100 ? "repeat(12, 1fr)" : (width >= 768 ? "repeat(2, 1fr)" : "1fr"),
                gap: 20,
                gridAutoFlow: "dense",
                position: "relative",
                zIndex: 10
              }}>
                {loading
                  ? Array.from({ length: 8 }).map((_, i) => (
                    <div key={i} style={{ gridColumn: width >= 1100 ? (i % 7 === 0 ? "span 6" : "span 3") : "span 1" }}>
                      <SkeletonCard index={i} isDark={dk} />
                    </div>
                  ))
                  : filtered.length === 0
                    ? <div style={{ gridColumn: "1/-1", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: "60px 0" }}>
                      <div style={{ fontSize: 36 }}>🔍</div>
                      <div style={{ fontSize: 14, color: dk ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.6)" }}>No results found</div>
                      <button onClick={() => { setQuery(""); setActiveCategory("All"); setActiveDiscovery("Trending"); }} style={{ padding: "8px 20px", borderRadius: 99, border: `1px solid ${A}55`, background: "none", color: A, fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Clear filters</button>
                    </div>
                    : filtered.map((tool, i) => {
                      const isFeatured = i % 7 === 0;
                      return (
                        <div key={tool.id} style={{
                          gridColumn: width >= 1100 ? (isFeatured ? "span 6" : "span 3") : "span 1",
                          cursor: "pointer"
                        }} onClick={() => setSelectedTool(tool)}>
                          <ToolCard tool={tool} index={i} isDark={dk} />
                        </div>
                      );
                    })
                }
              </div>
            </div>
            <div style={{ marginTop: 32, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, fontSize: 11, color: dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.5)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: A, animation: "pulseGlow 2s infinite" }} />
              Connected to AWS OpenSearch · Bedrock Semantic Search
              <Icon d={Icons.Wifi} size={11} color={dk ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.35)"} />
            </div>
          </div>
        </main>

        {/* Intent Discovery Sidebar widget (Desktop) */}
        {!isMobile && showChat && (
          <aside style={{
            width: 400, flexShrink: 0,
            borderLeft: `1px solid ${dk ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
            background: dk ? "rgba(2, 6, 23, 0.7)" : "rgba(255,255,255,0.7)",
            backdropFilter: "blur(24px)", zIndex: 60,
            display: "flex", flexDirection: "column"
          }}>
            <AIChatPanel onClose={() => setShowChat(false)} isDark={dk} isMobile={isMobile} />
          </aside>
        )}

        {/* Resource Detail Workbench Modal */}
        {selectedTool && (
          <ResourceWorkbench
            tool={selectedTool}
            onClose={() => setSelectedTool(null)}
            isDark={dk}
          />
        )}

        {/* Settings Modal */}
        {showSettings && (
          <SettingsModal
            isDark={dk}
            onClose={() => setShowSettings(false)}
            isHinglish={isHinglish}
            setIsHinglish={setIsHinglish}
            toggleTheme={toggleTheme}
            notifySystems={notifySystems}
            setNotifySystems={setNotifySystems}
          />
        )}

        {/* Auth Portal */}
        {showAuth && (
          <AuthPortal
            isDark={dk}
            onClose={() => setShowAuth(false)}
            onLogin={(u) => { setIsLoggedIn(true); setUser(u); setShowAuth(false); }}
          />
        )}

        {/* Resource Submission Modal */}
        {showSubmission && (
          <ResourceSubmissionModal
            isDark={dk}
            onClose={() => setShowSubmission(false)}
          />
        )}
      </div>
    </>
  );
}

