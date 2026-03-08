import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url);
        const limit = parseInt(searchParams.get('limit') || '20');
        const category = searchParams.get('category') || '';
        
        // Query database directly for trending resources
        // We replicate the backend's logic: sort by trending_score desc
        const resources = await prisma.resources.findMany({
            where: {
                ...(category ? { type: { contains: category, mode: 'insensitive' } } : {}),
                health_status: 'healthy'
            },
            orderBy: [
                { trending_score: 'desc' },
                { github_stars: 'desc' }
            ],
            take: limit,
        });

        return NextResponse.json({
            count: resources.length,
            resources: resources
        });
    } catch (error) {
        console.error("[API/trending] Error:", error);
        return NextResponse.json(
            { error: "Database query failed" },
            { status: 500 }
        );
    }
}
