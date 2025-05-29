#!/bin/bash
set -e

echo "🚀 Starting Production Deployment to Railway..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    print_error "Railway CLI is not installed. Install it with: npm install -g @railway/cli"
    exit 1
fi

# Check if logged in to Railway
if ! railway whoami &> /dev/null; then
    print_error "Not logged in to Railway. Run: railway login"
    exit 1
fi

print_status "✅ Railway CLI is installed and you're logged in"

# Check for required environment variables
print_status "🔍 Checking environment variables..."

REQUIRED_VARS=(
    "OPENAI_API_KEY"
    "ANTHROPIC_API_KEY"
    "SUPABASE_URL"
    "SUPABASE_SERVICE_KEY"
)

# Check if .env file exists and contains required variables
if [ -f .env ]; then
    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env; then
            print_status "✅ Found $var in .env"
        else
            print_warning "⚠️  $var not found in .env file"
        fi
    done
else
    print_warning "⚠️  No .env file found. Make sure environment variables are set in Railway dashboard"
fi

# Test local setup
print_status "🧪 Running quick health check..."
if [ -f "main.py" ]; then
    print_status "✅ main.py found"
else
    print_error "❌ main.py not found"
    exit 1
fi

if [ -f "requirements.txt" ]; then
    print_status "✅ requirements.txt found"
else
    print_error "❌ requirements.txt not found"
    exit 1
fi

# Ask for confirmation
echo ""
print_warning "⚠️  This will deploy your app to production. Make sure:"
print_warning "   1. All environment variables are set in Railway dashboard"
print_warning "   2. Supabase project is active and accessible"
print_warning "   3. All API keys are valid"
print_warning "   4. You've tested the app locally"
echo ""

read -p "Continue with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Deployment cancelled"
    exit 0
fi

# Deploy to Railway
print_status "🚀 Deploying to Railway..."

# Use railway deploy command
if railway deploy; then
    print_status "✅ Deployment successful!"
    
    # Get the deployment URL
    DEPLOY_URL=$(railway status --json | grep -o '"url":"[^"]*"' | cut -d'"' -f4)
    
    if [ ! -z "$DEPLOY_URL" ]; then
        print_status "🌐 Your app is deployed at: $DEPLOY_URL"
        print_status "🏥 Health check: $DEPLOY_URL/health"
        print_status "📚 API docs: $DEPLOY_URL/docs"
    fi
    
    print_status "🎉 Deployment completed successfully!"
else
    print_error "❌ Deployment failed!"
    print_error "Check the Railway logs for more details: railway logs"
    exit 1
fi

print_status "✅ Done! Monitor your app with: railway logs --follow" 