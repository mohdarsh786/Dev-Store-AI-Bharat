import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url);
        const limit = searchParams.get('limit') || '40';
        const category = searchParams.get('category') || '';
        const pricing_type = searchParams.get('pricing_type') || '';
        const sort = searchParams.get('sort') || '';
        
        // Use the correct backend endpoint with all parameters
        let url = `${BACKEND_URL}/api/v1/trending?limit=${limit}`;
        if (category && category !== 'All') {
            url += `&category=${encodeURIComponent(category)}`;
        }
        if (pricing_type) {
            url += `&pricing_type=${encodeURIComponent(pricing_type)}`;
        }
        if (sort) {
            url += `&sort=${encodeURIComponent(sort)}`;
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
