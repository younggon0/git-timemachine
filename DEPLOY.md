# Deployment Guide - Railway Web App

## Quick Deploy to Railway (Public Web Access)

### 1. Deploy to Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project (select "Deploy from GitHub repo" or "Deploy from local directory")
railway init

# Deploy your app
railway up
```

**Alternative: GitHub Integration**
1. Push your code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "Deploy Now" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-deploy from your main branch

### 2. Set Environment Variables
In your Railway dashboard, add these environment variables:
- `ANTHROPIC_API_KEY` - Your Anthropic API key
- `OPENAI_API_KEY` - Your OpenAI API key

### 3. Get Your Public URL
After deployment, Railway will provide a public URL like:
`https://your-app-name.up.railway.app`

### 4. Custom Domain (Optional)
To use your own domain:
1. Go to Railway dashboard → Settings → Domains
2. Add your custom domain
3. Update your DNS records as instructed

## Production Features Configured
- ✅ Headless mode for web deployment
- ✅ CORS disabled for public access
- ✅ Health check endpoint
- ✅ Auto-restart on failure
- ✅ Optimized for external web access
- ✅ Custom theme for professional look

## App Features
- 🔍 Semantic code analysis with AI
- 📊 Interactive visualizations
- ⏰ Git history timeline navigation
- 🤖 Natural language queries about code evolution

## Usage
1. Enter any public GitHub repository URL
2. Set your API keys in Railway environment variables
3. Share your Railway URL with anyone to use the app

Your app will be publicly accessible at the Railway-provided URL!