import { useState } from "react";

const POS = {
    right: { left: "calc(100% + 10px)", top: "50%", transform: "translateY(-50%)" },
    left: { right: "calc(100% + 10px)", top: "50%", transform: "translateY(-50%)" },
    bottom: { top: "calc(100% + 10px)", left: "50%", transform: "translateX(-50%)" },
    top: { bottom: "calc(100% + 10px)", left: "50%", transform: "translateX(-50%)" },
};

export default function Tooltip({ text, children, position = "right" }) {
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
                        background: "var(--ds-tooltip-bg)",
                        color: "var(--ds-tooltip-text)",
                        padding: "6px 12px",
                        borderRadius: 8,
                        fontSize: 11,
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        pointerEvents: "none",
                        zIndex: 9999,
                        animation: "fadeIn 0.15s ease",
                        boxShadow: "0 4px 16px rgba(0,0,0,0.25)",
                        letterSpacing: "0.01em",
                    }}
                >
                    {text}
                </div>
            )}
        </div>
    );
}
