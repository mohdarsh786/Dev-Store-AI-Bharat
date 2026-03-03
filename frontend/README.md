# DevStore Frontend

React-based frontend for the DevStore AI-powered developer marketplace with custom glassmorphism design.

## Features

- Custom glassmorphism design system (NO Tailwind, NO template libraries)
- Dark/Light theme support with smooth transitions
- Semantic search interface
- Category browsing
- Resource detail pages
- Solution blueprint visualization
- Multilingual support (English, Hindi, Hinglish, Tamil, Telugu, Bengali)
- Boilerplate code generation

## Setup

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running Locally

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The app will be available at `http://localhost:3000`

### Running Tests

```bash
# Run tests
npm test

# Run tests with UI
npm run test:ui
```

## Project Structure

```
frontend/
├── index.html           # HTML entry point
├── vite.config.js       # Vite configuration
├── package.json         # Dependencies
├── src/
│   ├── main.jsx        # React entry point
│   ├── App.jsx         # Root component
│   ├── components/     # Reusable components
│   ├── pages/          # Page components
│   ├── services/       # API clients
│   ├── utils/          # Utility functions
│   ├── hooks/          # Custom React hooks
│   └── styles/         # CSS files (vanilla CSS with CSS Modules)
│       ├── index.css   # Global styles and CSS variables
│       └── App.css     # App-specific styles
└── public/             # Static assets
```

## Design System

### Theme

The app uses CSS custom properties for theming:

- **Light Mode**: White (#FFFFFF) + Blue (#0052CC)
- **Dark Mode**: Black (#000000) + Blue (#0066FF)

### Glassmorphism Components

All UI components use custom glassmorphism styling:
- Semi-transparent backgrounds with backdrop blur
- Subtle borders with blue tint
- Smooth hover effects with increased opacity
- Consistent border-radius (12-20px)

### CSS Architecture

- **CSS Modules**: Component-scoped styles
- **CSS Custom Properties**: Theme variables in `:root`
- **No frameworks**: Pure vanilla CSS implementation

## Deployment

### Build for Production

```bash
npm run build
```

### Deploy to S3

```bash
# Sync build to S3
aws s3 sync dist/ s3://devstore-frontend-prod/

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id YOUR_DIST_ID \
  --paths "/*"
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

MIT
