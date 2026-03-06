import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url);
        const params = searchParams.toString();
        const url = `${BACKEND_URL}/api/v1/resources${params ? `?${params}` : ""}`;

        const response = await fetch(url, {
            headers: { "Content-Type": "application/json" },
            next: { revalidate: 30 }, // ISR: cache for 30s
        });

        if (!response.ok) {
            return NextResponse.json(
                { error: `Backend error: ${response.status}` },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("[API/resources] Error:", error);
        return NextResponse.json(
            { error: "Backend unreachable" },
            { status: 503 }
        );
    }
}
