import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url);
        const q = searchParams.get('q') || '';
        const category = searchParams.get('category') || '';
        const limit = parseInt(searchParams.get('limit') || '20');
        const offset = parseInt(searchParams.get('offset') || '0');
        
        // Direct Prisma search
        const resources = await prisma.resources.findMany({
            where: {
                AND: [
                    q ? {
                        OR: [
                            { name: { contains: q, mode: 'insensitive' } },
                            { description: { contains: q, mode: 'insensitive' } }
                        ]
                    } : {},
                    category ? { type: { contains: category, mode: 'insensitive' } } : {}
                ]
            },
            orderBy: {
                github_stars: 'desc'
            },
            take: limit,
            skip: offset
        });

        const total = await prisma.resources.count({
            where: {
                AND: [
                    q ? {
                        OR: [
                            { name: { contains: q, mode: 'insensitive' } },
                            { description: { contains: q, mode: 'insensitive' } }
                        ]
                    } : {},
                    category ? { type: { contains: category, mode: 'insensitive' } } : {}
                ]
            }
        });

        return NextResponse.json({
            resources,
            total,
            limit,
            offset
        });
    } catch (error) {
        console.error("[API/resources] Error:", error);
        return NextResponse.json(
            { error: "Database query failed" },
            { status: 500 }
        );
    }
}
