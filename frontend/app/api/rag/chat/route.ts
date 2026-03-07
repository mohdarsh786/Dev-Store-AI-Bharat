import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { query, session_id = "default", filters = {} } = body;

        if (!query || typeof query !== "string") {
            return NextResponse.json(
                { error: "Query is required and must be a string" },
                { status: 400 }
            );
        }

        // Forward request to FastAPI backend
        const response = await fetch(`${BACKEND_URL}/api/v1/rag/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                query,
                session_id,
                filters,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Backend RAG chat error:", errorText);
            return NextResponse.json(
                { error: "RAG chat request failed", details: errorText },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("RAG chat route error:", error);
        return NextResponse.json(
            { error: "Internal server error", details: String(error) },
            { status: 500 }
        );
    }
}
