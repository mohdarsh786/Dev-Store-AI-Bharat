import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { query = '' } = body;
        
        // Direct Prisma keyword search for parity with simple keyword backend
        const resources = await prisma.resources.findMany({
            where: {
                OR: [
                    { name: { contains: query, mode: 'insensitive' } },
                    { description: { contains: query, mode: 'insensitive' } }
                ]
            },
            take: 20
        });

        return NextResponse.json({
            resources
        });
    } catch (error) {
        console.error("[API/search] Error:", error);
        return NextResponse.json(
            { error: "Database search failed" },
            { status: 500 }
        );
    }
}
