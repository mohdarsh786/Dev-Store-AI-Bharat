import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/v1/health`, {
            next: { revalidate: 10 },
        });

        if (!response.ok) {
            return NextResponse.json({ status: "offline" }, { status: 503 });
        }

        const data = await response.json();
        return NextResponse.json({ status: "online", ...data });
    } catch {
        return NextResponse.json({ status: "offline" }, { status: 503 });
    }
}
