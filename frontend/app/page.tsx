import DevStoreDashboard from "@/components/DevStoreDashboard";

// This page is intentionally not a Server Component that fetches data —
// the DevStoreDashboard is a full client component with SWR-based fetching.
// SEO metadata is handled by layout.tsx.
export default function HomePage() {
  return <DevStoreDashboard />;
}
