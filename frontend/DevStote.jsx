// Dev-Store Dashboard — upgraded from uploaded code
// Drop this into app/page.tsx (rename + add "use client") for Next.js
// Or use as a standalone React component

import { useState, useEffect, useCallback, useRef } from "react";

// ─── Lucide-style SVG Icons (self-contained, no import needed in plain React) ──
const Icon = ({ d, size = 16, color = "currentColor", strokeWidth = 1.75 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
        {Array.isArray(d) ? d.map((path, i) => <path key={i} d={path} />) : <path d={d} />}
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
};

// ─── TypeScript-compatible Tool interface (JS object structure) ───────────────
// interface Tool {
//   id: string; name: string; category: string; description: string;
//   installCommand: string; status: 'stable' | 'down';
//   stars: number; latency: number; iconEmoji: string; accentColor: string;
// }

const MOCK_TOOLS = [
    { id: "stripe", name: "Stripe", category: "Payments", description: "Full-stack payments infrastructure for the internet", installCommand: "npm install stripe @stripe/stripe-js", status: "stable", stars: 3600, latency: 42, iconEmoji: "💳", accentColor: "#635BFF" },
    { id: "supabase", name: "Supabase", category: "Database", description: "Open source Firebase alternative built on Postgres", installCommand: "npm install @supabase/supabase-js", status: "stable", stars: 71000, latency: 28, iconEmoji: "⚡", accentColor: "#3ECF8E" },
    { id: "docker", name: "Docker", category: "DevOps", description: "Containerize and ship applications anywhere, reliably", installCommand: "curl -fsSL https://get.docker.com | sh", status: "stable", stars: 28000, latency: 5, iconEmoji: "🐳", accentColor: "#2496ED" },
    { id: "redis", name: "Redis", category: "Cache", description: "In-memory data structure store and blazing-fast cache", installCommand: "npm install ioredis", status: "stable", stars: 66000, latency: 0.8, iconEmoji: "🔴", accentColor: "#FF4438" },
    { id: "prisma", name: "Prisma", category: "ORM", description: "Next-gen TypeScript ORM with type-safe database queries", installCommand: "npm install prisma @prisma/client", status: "stable", stars: 38000, latency: 12, iconEmoji: "🔷", accentColor: "#5A67D8" },
    { id: "razorpay", name: "Razorpay", category: "Payments", description: "India's leading gateway — UPI, cards, net banking, wallets", installCommand: "npm install razorpay", status: "stable", stars: 1200, latency: 55, iconEmoji: "🇮🇳", accentColor: "#3395FF" },
    { id: "upstash", name: "Upstash", category: "Serverless", description: "Serverless Redis and Kafka with per-request pricing", installCommand: "npm install @upstash/redis", status: "stable", stars: 4800, latency: 18, iconEmoji: "☁️", accentColor: "#00E9A3" },
    { id: "mongodb", name: "MongoDB", category: "Database", description: "Flexible document database for modern applications", installCommand: "npm install mongoose mongodb", status: "down", stars: 26000, latency: 34, iconEmoji: "🍃", accentColor: "#47A248" },
];

const CATEGORIES = ["All", "Payments", "Database", "DevOps", "Cache", "ORM", "Serverless"];

const GHOST_TEXTS = [
    "Bhai, best payment gateway batao...",
    "Looking for a NoSQL Database?",
    "Yaar, Docker se kaise deploy karein?",
    "Find low-latency caching solutions...",
    "Kaunsa ORM use karein 2025 mein?",
    "Best serverless database for Next.js?",
];

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
        const SPEEDS = { typing: 46, erasing: 22, pause: 2200, gap: 380 };

        const tick = (now) => {
            const target = GHOST_TEXTS[idx.current];
            const delay =
                phase.current === "typing" ? SPEEDS.typing :
                    phase.current === "erasing" ? SPEEDS.erasing :
                        phase.current === "pause" ? SPEEDS.pause : SPEEDS.gap;

            if (now - last >= delay) {
                last = now;
                if (phase.current === "typing") {
                    if (chars.current < target.length) {
                        chars.current++;
                        setText(target.slice(0, chars.current));
                    } else { phase.current = "pause"; }
                } else if (phase.current === "pause") {
                    phase.current = "erasing";
                } else if (phase.current === "erasing") {
                    if (chars.current > 0) {
                        chars.current--;
                        setText(target.slice(0, chars.current));
                    } else {
                        idx.current = (idx.current + 1) % GHOST_TEXTS.length;
                        phase.current = "gap";
                    }
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
    const [copied, setCopied] = useState(false);
    const [hovered, setHovered] = useState(false);

    const handleCopy = async () => {
        try { await navigator.clipboard.writeText(tool.installCommand); } catch { }
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);

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
                display: "flex",
                flexDirection: "column",
                gap: 14,
                position: "relative",
                overflow: "hidden",
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
                        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.32)", marginTop: 2, fontWeight: 500 }}>
                            {tool.category}
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
                {tool.description}
            </p>

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

            {/* Install command box */}
            <div style={{
                background: "rgba(0,0,0,0.45)", borderRadius: 14,
                padding: "10px 14px",
                border: "1px solid rgba(255,255,255,0.055)",
                backdropFilter: "blur(4px)",
            }}>
                <code style={{
                    fontFamily: "'Fira Code', 'JetBrains Mono', monospace",
                    fontSize: 11.5,
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
                    width: "100%", border: "none", borderRadius: 99,
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
                    transform: "scale(1)",
                    transition: "all 0.25s ease",
                    letterSpacing: "-0.01em",
                }}
                onMouseDown={e => e.currentTarget.style.transform = "scale(0.97)"}
                onMouseUp={e => e.currentTarget.style.transform = "scale(1)"}
            >
                <Icon d={copied ? Icons.Check : Icons.Copy} size={14} color={copied ? "#00FFA3" : "rgba(255,255,255,0.6)"} />
                {copied ? "Copied!" : "Copy Install Command"}
            </button>
        </div>
    );
}

// ─── Skeleton Card ─────────────────────────────────────────────────────────────
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
            <Icon d={Icons[iconKey]} size={18} color={active ? "#00FFA3" : hov ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.28)"} />
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

// ─── Main Dashboard ────────────────────────────────────────────────────────────
export default function DevStore() {
    const [tools, setTools] = useState([]);
    const [filtered, setFiltered] = useState([]);
    const [query, setQuery] = useState("");
    const [activeCategory, setActiveCategory] = useState("All");
    const [activeNav, setActiveNav] = useState(0);
    const [loading, setLoading] = useState(true);
    const [focused, setFocused] = useState(false);
    const debounceRef = useRef(null);
    const ghostText = useGhostText(!focused && !query);

    // Simulate AWS Lambda fetch
    useEffect(() => {
        const t = setTimeout(() => {
            setTools(MOCK_TOOLS);
            setFiltered(MOCK_TOOLS);
            setLoading(false);
        }, 750);
        return () => clearTimeout(t);
    }, []);

    // Debounced search — simulates Lambda@Edge query
    const runFilter = useCallback((q, cat, data) => {
        clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => {
            const lq = q.toLowerCase().trim();
            setFiltered(data.filter(t => {
                const catOk = cat === "All" || t.category === cat;
                const qOk = !lq ||
                    t.name.toLowerCase().includes(lq) ||
                    t.description.toLowerCase().includes(lq) ||
                    t.category.toLowerCase().includes(lq);
                return catOk && qOk;
            }));
        }, 220);
    }, []);

    useEffect(() => { runFilter(query, activeCategory, tools); }, [query, activeCategory, tools]);

    const stableCount = tools.filter(t => t.status === "stable").length;
    const NAV = [
        { key: "Home", label: "Home" },
        { key: "Cpu", label: "APIs" },
        { key: "Database", label: "Data" },
        { key: "Layers", label: "Stack" },
    ];

    return (
        <>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        body { overflow:hidden; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.08); border-radius:4px; }
        @keyframes cardIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulseGlow { 0%,100% { opacity:1; box-shadow:0 0 6px currentColor; } 50% { opacity:0.4; box-shadow:none; } }
        @keyframes shimmer { 0%,100%{opacity:0.45} 50%{opacity:0.8} }
        @keyframes fadeIn { from{opacity:0} to{opacity:1} }
        .cat-btn:hover { background:rgba(255,255,255,0.07) !important; }
        input::placeholder { color:rgba(255,255,255,0.2); font-style:italic; transition:color 0.2s; }
        input:focus { outline:none; }
        button:focus { outline:none; }
      `}</style>

            <div style={{
                display: "flex", height: "100vh", width: "100vw", overflow: "hidden",
                background: "#0C0C0E", color: "#fff",
                fontFamily: "'DM Sans', sans-serif",
            }}>
                {/* ── Sidebar / Navigation Rail ─────────────────────────────── */}
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
                        boxShadow: "0 4px 20px rgba(0,255,163,0.25)",
                    }}>
                        D
                    </div>

                    {/* Nav */}
                    <div style={{ width: "100%", padding: "0 12px", display: "flex", flexDirection: "column", gap: 4 }}>
                        {NAV.map((n, i) => (
                            <NavItem key={n.key} iconKey={n.key} label={n.label}
                                active={activeNav === i} onClick={() => setActiveNav(i)} />
                        ))}
                    </div>

                    {/* Bottom */}
                    <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
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
                        }}>A</div>
                    </div>
                </aside>

                {/* ── Main Column ───────────────────────────────────────────── */}
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
                                Dev-Store
                            </div>
                            <div style={{ fontSize: 15, fontWeight: 700, color: "rgba(255,255,255,0.82)", letterSpacing: "-0.02em" }}>
                                Tools & Integrations
                            </div>
                        </div>

                        {/* Search */}
                        <div style={{ flex: 1, maxWidth: 640, margin: "0 auto", position: "relative" }}>
                            <div style={{ position: "absolute", left: 16, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}>
                                <Icon d={Icons.Search} size={15} color="rgba(255,255,255,0.22)" />
                            </div>
                            <input
                                type="text"
                                value={query}
                                onChange={e => setQuery(e.target.value)}
                                onFocus={() => setFocused(true)}
                                onBlur={() => setFocused(false)}
                                placeholder={focused ? "Search tools, categories, commands..." : ghostText}
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
                        </div>

                        {/* Stat pills */}
                        <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
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

                    {/* ── Category Pill Bar ──────────────────────────────────── */}
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
                                onClick={() => setActiveCategory(cat)}
                                style={{
                                    padding: "7px 16px", borderRadius: 99,
                                    fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12,
                                    cursor: "pointer", flexShrink: 0, transition: "all 0.2s",
                                    background: activeCategory === cat ? "#006B5D" : "rgba(255,255,255,0.04)",
                                    color: activeCategory === cat ? "#fff" : "rgba(255,255,255,0.38)",
                                    border: `1px solid ${activeCategory === cat ? "#006B5D" : "rgba(255,255,255,0.06)"}`,
                                }}
                            >
                                {cat}
                            </button>
                        ))}
                        <div style={{ marginLeft: "auto", flexShrink: 0, display: "flex", alignItems: "center", gap: 5, fontSize: 11.5, color: "rgba(255,255,255,0.2)" }}>
                            <Icon d={Icons.ChevronR} size={13} />
                            {loading ? "—" : filtered.length} result{filtered.length !== 1 ? "s" : ""}
                        </div>
                    </div>

                    {/* ── Bento Grid ────────────────────────────────────────────── */}
                    <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
                        <div style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(4, 1fr)",
                            gap: 16,
                        }}>
                            {loading
                                ? Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} index={i} />)
                                : filtered.length === 0
                                    ? (
                                        <div style={{ gridColumn: "1/-1", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: "60px 0" }}>
                                            <div style={{ fontSize: 36 }}>🔍</div>
                                            <div style={{ fontSize: 14, color: "rgba(255,255,255,0.3)" }}>No tools found for <em>"{query}"</em></div>
                                            <button onClick={() => { setQuery(""); setActiveCategory("All"); }}
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
                            Connected to AWS DynamoDB · Lambda@Edge queries
                            <Icon d={Icons.Wifi} size={11} color="rgba(255,255,255,0.15)" />
                        </div>
                    </div>
                </main>
            </div>
        </>
    );
}
