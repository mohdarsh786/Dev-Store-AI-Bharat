import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { query = '' } = body;

        // Simplified intent logic using direct database check
        const topMatches = await prisma.resources.findMany({
            where: {
                OR: [
                    { name: { contains: query, mode: 'insensitive' } },
                    { description: { contains: query, mode: 'insensitive' } }
                ]
            },
            take: 5
        });

        return NextResponse.json({
            intent: "search",
            query,
            suggested_filters: {},
            results: topMatches
        });
    } catch (error) {
        console.error("[API/search/intent] Error:", error);
        return NextResponse.json({ intent: "general", query: "" });
    }
}
