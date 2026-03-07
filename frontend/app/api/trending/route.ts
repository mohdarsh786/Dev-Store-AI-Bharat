import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url);
        const limit = searchParams.get('limit') || '20';
        const category = searchParams.get('category') || '';
        
        // Use the correct backend endpoint
        let url = `${BACKEND_URL}/api/resources/trending?limit=${limit}`;
        if (category) {
            url += `&category=${encodeURIComponent(category)}`;
        }

        const response = await fetch(url, {
            headers: { "Content-Type": "application/json" },
            next: { revalidate: 60 }, // ISR: cache for 60s (trending changes less often)
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
        console.error("[API/trending] Error:", error);
        return NextResponse.json(
            { error: "Backend unreachable" },
            { status: 503 }
        );
    }
}
