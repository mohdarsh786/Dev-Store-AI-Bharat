import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DevStore — AI for Bharat | Developer Marketplace",
  description:
    "Discover, compare, and integrate AI APIs, open-source models, and datasets curated for Indian developers. Powered by AWS Bedrock and OpenSearch semantic search.",
  keywords: [
    "AI marketplace", "developer tools", "India", "Bharat", "OpenAI", "Gemini",
    "Llama", "Mistral", "ML models", "datasets", "API catalog",
  ],
  authors: [{ name: "DevStore AI Bharat Team" }],
  openGraph: {
    title: "DevStore — AI for Bharat",
    description: "The premier AI developer marketplace for Bharat",
    type: "website",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
