#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🔍 DevStore Frontend Setup Check\n');

const checks = [
  { name: 'index.html', path: 'index.html' },
  { name: 'src/main.jsx', path: 'src/main.jsx' },
  { name: 'src/App.jsx', path: 'src/App.jsx' },
  { name: 'src/pages/TrinityDashboard.jsx', path: 'src/pages/TrinityDashboard.jsx' },
  { name: 'src/styles/trinity.css', path: 'src/styles/trinity.css' },
  { name: 'src/hooks/useSearch.js', path: 'src/hooks/useSearch.js' },
  { name: 'src/hooks/useResources.js', path: 'src/hooks/useResources.js' },
  { name: 'src/services/api.js', path: 'src/services/api.js' },
  { name: 'src/components/SearchResultCard.jsx', path: 'src/components/SearchResultCard.jsx' },
  { name: 'package.json', path: 'package.json' },
  { name: 'vite.config.js', path: 'vite.config.js' }
];

let allGood = true;

checks.forEach(check => {
  const filePath = path.join(__dirname, check.path);
  const exists = fs.existsSync(filePath);
  const status = exists ? '✅' : '❌';
  console.log(`${status} ${check.name}`);
  if (!exists) allGood = false;
});

console.log('\n' + (allGood ? '✅ All files present!' : '❌ Some files missing!'));
console.log('\nTo start frontend:');
console.log('  npm install');
console.log('  npm run dev');
