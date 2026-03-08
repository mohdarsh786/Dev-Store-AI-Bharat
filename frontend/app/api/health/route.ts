import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET() {
    try {
        // Basic connectivity check to the database
        await prisma.$queryRaw`SELECT 1`;
        
        return NextResponse.json({ 
            status: "online", 
            service: "devstore-api-v2",
            database: "connected"
        });
    } catch (error) {
        console.error("[API/health] Error:", error);
        return NextResponse.json({ 
            status: "offline", 
            database: "disconnected" 
        }, { status: 503 });
    }
}
