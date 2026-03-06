"use client";

import { useState } from "react";

const POS: Record<string, React.CSSProperties> = {
    right: { left: "calc(100% + 10px)", top: "50%", transform: "translateY(-50%)" },
    left: { right: "calc(100% + 10px)", top: "50%", transform: "translateY(-50%)" },
    bottom: { top: "calc(100% + 10px)", left: "50%", transform: "translateX(-50%)" },
    top: { bottom: "calc(100% + 10px)", left: "50%", transform: "translateX(-50%)" },
};

interface TooltipProps {
    text: string;
    children: React.ReactNode;
    position?: "right" | "left" | "bottom" | "top";
}

export default function Tooltip({ text, children, position = "right" }: TooltipProps) {
    const [show, setShow] = useState(false);

    return (
        <div
            style={{ position: "relative", display: "inline-flex" }}
            onMouseEnter={() => setShow(true)}
            onMouseLeave={() => setShow(false)}
        >
            {children}
            {show && (
                <div
                    style={{
                        position: "absolute",
                        ...POS[position],
                        background: "rgba(10, 16, 32, 0.92)",
                        color: "rgba(255,255,255,0.9)",
                        padding: "6px 12px",
                        borderRadius: 8,
                        fontSize: 11,
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        pointerEvents: "none",
                        zIndex: 9999,
                        animation: "fadeIn 0.15s ease",
                        boxShadow: "0 4px 16px rgba(0,0,0,0.35)",
                        letterSpacing: "0.01em",
                        border: "1px solid rgba(255,255,255,0.08)",
                    }}
                >
                    {text}
                </div>
            )}
        </div>
    );
}
