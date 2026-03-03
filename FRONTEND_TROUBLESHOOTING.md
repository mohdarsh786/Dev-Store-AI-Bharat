# Frontend Troubleshooting

## Common Issues & Fixes

### Issue 1: Port Already in Use
**Error:** `EADDRINUSE: address already in use :::5173`

**Fix:**
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9

# Or use different port
npm run dev -- --port 3000
```

### Issue 2: Module Not Found
**Error:** `Cannot find module`

**Fix:**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Issue 3: CSS Not Loading
**Error:** Styles not applied

**Fix:**
- Ensure trinity.css exists in `frontend/src/styles/`
- Check CSS import in App.jsx and TrinityDashboard.jsx
- Clear browser cache (Ctrl+Shift+Delete)

### Issue 4: API Connection Failed
**Error:** Search returns no results

**Fix:**
```bash
# Check backend is running
curl http://localhost:8000/api/v1/health

# If not running, start backend
cd backend
python -m uvicorn main:app --reload --port 8000
```

### Issue 5: Blank Page
**Error:** White/blank screen

**Fix:**
```bash
# Check browser console for errors (F12)
# Try hard refresh (Ctrl+Shift+R)
# Check if React is loading
npm run dev
```

## Quick Start

```bash
# Terminal 1 - Backend
cd backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

Then open: http://localhost:5173

## Verify Setup

1. Backend health: http://localhost:8000/api/v1/health
2. Frontend: http://localhost:5173
3. Search should work with mock data
