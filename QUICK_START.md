# Quick Start Guide - DevStore

## 🚀 Start in 5 Minutes

### Backend
```bash
cd Dev-Store-AI-Bharat/backend
.\venv\Scripts\Activate.ps1  # Windows
uvicorn api_gateway:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd Dev-Store-AI-Bharat/frontend
npm install
npm run dev
```

### Access
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

---

## 📚 Full Documentation

- **COMPLETE_DEPLOYMENT_GUIDE.md** - Step-by-step deployment (all paths)
- **DEPLOYMENT_READINESS.md** - Deployment status and checklist
- **VERIFICATION_COMPLETE.md** - Code verification results
- **FIXES_APPLIED.md** - List of all fixes made
- **AWS_SETUP_GUIDE.md** - AWS infrastructure setup
- **EC2_DEPLOYMENT_GUIDE.md** - EC2 deployment details

---

## ✅ Current Status

### Code Quality
- ✅ All critical bugs fixed
- ✅ All dependencies installed
- ✅ All verification tests passed
- ✅ Ready for deployment

### Deployment Options
1. **Local Demo** - ✅ Ready NOW (5 minutes)
2. **Local + AWS** - ⚠️ Needs AWS setup (2-3 hours)
3. **Production EC2** - ⚠️ Needs full infrastructure (4-6 hours)

---

## 🔧 Common Commands

### Backend
```bash
# Start development server
uvicorn api_gateway:app --reload

# Start production server
gunicorn api_gateway:app -w 4 -k uvicorn.workers.UvicornWorker

# Run migrations
python run_migrations.py

# Update rankings
python update_rankings.py

# Verify fixes
python verify_fixes.py
```

### Frontend
```bash
# Development
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run tests
npm test
```

### Deployment
```bash
# Deploy to EC2 (from EC2 instance)
./deploy.sh

# Check service status
sudo systemctl status devstore-api

# View logs
sudo journalctl -u devstore-api -f
```

---

## 🆘 Quick Troubleshooting

### Backend won't start
```bash
# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Check logs
python -c "import api_gateway"
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Database connection error
```bash
# Check .env file
cat backend/.env | grep DATABASE_URL

# Test connection
python -c "from clients.database import DatabaseClient; import asyncio; asyncio.run(DatabaseClient().connect())"
```

### AWS services not working
```bash
# Check environment variables
python -c "from config import settings; print(settings.opensearch_host)"

# Verify AWS credentials
aws sts get-caller-identity
```

---

## 📞 Need Help?

1. Check **COMPLETE_DEPLOYMENT_GUIDE.md** for detailed instructions
2. Check **DEPLOYMENT_READINESS.md** for deployment status
3. Check **VERIFICATION_COMPLETE.md** for code verification
4. Review error logs in `/var/log/devstore/` (production)
5. Check systemd logs: `sudo journalctl -u devstore-api`

---

## 🎯 Next Steps

### For Testing/Demo
1. Start backend and frontend (commands above)
2. Open http://localhost:5173
3. Try searching and browsing

### For Production
1. Follow **COMPLETE_DEPLOYMENT_GUIDE.md** - Path 3
2. Set up AWS infrastructure
3. Deploy to EC2
4. Configure domain and SSL
5. Monitor and maintain

---

## 💡 Tips

- Use mock data mode for quick testing (no AWS needed)
- Set up AWS services incrementally (database → redis → opensearch → bedrock)
- Monitor logs regularly: `sudo journalctl -u devstore-api -f`
- Keep backups of your database
- Use the deployment script for updates: `./deploy.sh`

---

**Ready to start?** Run the commands at the top of this file! 🚀
