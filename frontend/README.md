# DevStore Frontend: AI-Powered Discovery Interface

The frontend for DevStore is a high-performance **Next.js 15** application designed for the Indian developer ecosystem. It features a modern, responsive UI with Glassmorphism aesthetics and native support for multilingual interactions (English, Hindi, Hinglish).

---

## 🚀 Tech Stack

- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS + Glassmorphism Tokens
- **State Management:** SWR for high-performance data fetching
- **Deployment:** AWS Amplify

## 🏗️ Architecture

The frontend serves as a thin, secure client layer that proxies all sensitive AI and database operations through **Next.js Route Handlers**.

- **App Router:** Utilizes Server Components for SEO and Client Components for interactivity.
- **Route Handlers (`/api/*`):** Securely forwards requests to the FastAPI backend, ensuring AWS and Database credentials never reach the browser.
- **Responsive Design:** Optimized for both desktop and mobile developers across Bharat.

## 📂 Structure

```text
frontend/
├── app/
│   ├── api/                # Secure server-side proxies to FastAPI
│   ├── layout.tsx          # Global layouts & SEO metadata
│   └── page.tsx            # Main DevStore Dashboard entry
├── components/
│   ├── dev-store/          # Core dashboard components
│   ├── ui/                 # Reusable UI primitives (Bento Grid, Cards)
│   └── shared/             # Common layouts (Navbar, Footer)
├── lib/
│   ├── api.ts              # SWR fetchers and API client logic
│   └── utils.ts            # Formatting and helper functions
├── public/                 # Static assets and icons
└── next.config.ts          # Build-time configuration
```

## 🛠️ Getting Started

### Prerequisites
- Node.js 18.17+
- Backend running at `localhost:8000` (or configured production URL)

### Installation
1. **Install Dependencies:**
   ```bash
   npm install
   ```

2. **Configure Environment:**
   ```bash
   cp .env.local.example .env.local
   # Set BACKEND_URL=http://localhost:8000
   ```

3. **Development Mode:**
   ```bash
   npm run dev
   ```

## ✨ Key UI Features

- **Hinglish Search Bar:** Intent discovery that handles natural language queries like *"Bhai, best payment gateway batao"*.
- **Trinity Dashboard:** A Bento Grid layout showing Trending, Recently Updated, and Highly Rated resources.
- **Multilingual RAG Chat:** A floating assistant that provides context-aware resource recommendations.
- **Glassmorphic Cards:** High-fidelity components showing stars, downloads, and pricing badges.

## 📜 Deployment

The frontend is optimized for **AWS Amplify**. 
- Push to the `main` branch to trigger the CI/CD pipeline.
- Custom domain and SSL are handled automatically by the Amplify CDN.

**License:** Apache License 2.0
