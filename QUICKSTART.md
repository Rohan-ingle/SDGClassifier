# 🚀 Quick Start Guide - SDG Classifier Deployment

This guide provides multiple deployment options for the SDG Classifier application.

## 📋 Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Azure CLI (for cloud deployment)
- Git

## 🎯 Deployment Options

### Option 1: Local Development (Fastest)

**Perfect for**: Testing, development, quick demos

```bash
# Clone repository
git clone https://github.com/Rohan-ingle/SDGClassifier.git
cd SDGClassifier

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Access at: http://localhost:8501
```

**Time to deploy**: ~2-3 minutes

---

### Option 2: Docker (Recommended for Production)

**Perfect for**: Consistent environments, easy deployment

```bash
# Build Docker image
docker build -t sdg-classifier:latest .

# Run container
docker run -p 8501:8501 sdg-classifier:latest

# Access at: http://localhost:8501
```

**Time to deploy**: ~5 minutes

**With Docker Compose** (if you have docker-compose.yml):
```bash
docker-compose up -d
```

---

### Option 3: Azure Kubernetes Service (Scalable Cloud)

**Perfect for**: Production, auto-scaling, high availability

#### A. Quick Setup (Using Script)

```bash
# Make deployment script executable
chmod +x deploy-aks.sh

# Run interactive menu
./deploy-aks.sh

# Select option 1: Full Setup
```

**Time to deploy**: ~15-20 minutes

#### B. Manual Setup

**Step 1: Azure Setup** (One-time)

```bash
# Login to Azure
az login

# Create resources
RESOURCE_GROUP="sdg-classifier-rg"
LOCATION="eastus"
ACR_NAME="sdgclassifieracr"
AKS_NAME="sdg-classifier-aks"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Container Registry
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Standard

# Create AKS cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_NAME \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_NAME
```

**Step 2: Build & Push Image**

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build and push (using ACR build - recommended)
az acr build --registry $ACR_NAME --image sdg-classifier:latest .

# OR build locally and push
docker build -t $ACR_NAME.azurecr.io/sdg-classifier:latest .
docker push $ACR_NAME.azurecr.io/sdg-classifier:latest
```

**Step 3: Deploy to Kubernetes**

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Update deployment with your ACR name
export ACR_NAME=$ACR_NAME
export IMAGE_TAG=latest
envsubst < k8s/deployment.yaml | kubectl apply -f - -n sdg-classifier

# Apply auto-scaling
kubectl apply -f k8s/hpa.yaml

# Get external IP (wait a few minutes)
kubectl get service sdg-classifier-service -n sdg-classifier
```

**Step 4: Access Application**

```bash
# Get the external IP
EXTERNAL_IP=$(kubectl get service sdg-classifier-service -n sdg-classifier -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Access your application at: http://$EXTERNAL_IP"
```

**Time to deploy**: ~20-30 minutes (first time)

---

### Option 4: GitHub Actions CI/CD (Automated)

**Perfect for**: Continuous deployment, team collaboration

#### Setup GitHub Secrets

1. Go to your repository → Settings → Secrets and variables → Actions
2. Add these secrets:

```
AZURE_CREDENTIALS     - Service principal JSON
ACR_NAME             - Your ACR name (e.g., sdgclassifieracr)
AKS_CLUSTER_NAME     - Your AKS name (e.g., sdg-classifier-aks)
AKS_RESOURCE_GROUP   - Your resource group (e.g., sdg-classifier-rg)
```

#### Create Service Principal

```bash
az ad sp create-for-rbac \
  --name "sdg-classifier-github-sp" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/sdg-classifier-rg \
  --sdk-auth
```

Copy the entire JSON output and add as `AZURE_CREDENTIALS` secret.

#### Trigger Deployment

```bash
# Push to main/master branch
git add .
git commit -m "Deploy to AKS"
git push origin main
```

The GitHub Actions workflow will automatically:
1. ✅ Run tests
2. ✅ Build Docker image
3. ✅ Push to ACR
4. ✅ Deploy to AKS
5. ✅ Run health checks

Monitor progress at: `https://github.com/<your-username>/SDGClassifier/actions`

---

## 📊 Monitoring & Management

### Check Deployment Status

```bash
# Kubernetes
kubectl get all -n sdg-classifier
kubectl get pods -n sdg-classifier
kubectl get hpa -n sdg-classifier

# View logs
kubectl logs -l app=sdg-classifier -n sdg-classifier --tail=50
```

### Access Application

**Local**: http://localhost:8501
**Docker**: http://localhost:8501
**AKS**: http://`<EXTERNAL_IP>`

### Test the Application

```bash
# Get external IP for AKS
EXTERNAL_IP=$(kubectl get service sdg-classifier-service -n sdg-classifier -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test endpoint
curl http://$EXTERNAL_IP

# Or open in browser
echo "Open in browser: http://$EXTERNAL_IP"
```

---

## 🛠️ Common Commands

### Docker Commands
```bash
# Build
docker build -t sdg-classifier:latest .

# Run
docker run -p 8501:8501 sdg-classifier:latest

# Stop all containers
docker stop $(docker ps -a -q)

# Remove all containers
docker rm $(docker ps -a -q)

# View logs
docker logs <container-id>
```

### Kubernetes Commands
```bash
# Get all resources
kubectl get all -n sdg-classifier

# Describe deployment
kubectl describe deployment sdg-classifier-deployment -n sdg-classifier

# Scale manually
kubectl scale deployment sdg-classifier-deployment --replicas=5 -n sdg-classifier

# Restart deployment
kubectl rollout restart deployment sdg-classifier-deployment -n sdg-classifier

# Port forward for local access
kubectl port-forward service/sdg-classifier-service 8501:80 -n sdg-classifier

# Delete deployment
kubectl delete namespace sdg-classifier
```

### Azure Commands
```bash
# List all resources
az resource list --resource-group sdg-classifier-rg --output table

# Stop AKS (save costs)
az aks stop --resource-group sdg-classifier-rg --name sdg-classifier-aks

# Start AKS
az aks start --resource-group sdg-classifier-rg --name sdg-classifier-aks

# Delete all resources
az group delete --name sdg-classifier-rg --yes --no-wait
```

---

## 🐛 Troubleshooting

### Issue: Port 8501 already in use

```bash
# Find and kill process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app.py --server.port 8502
```

### Issue: Docker image fails to build

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t sdg-classifier:latest .
```

### Issue: Pods not starting in AKS

```bash
# Check pod status
kubectl describe pod <pod-name> -n sdg-classifier

# View events
kubectl get events -n sdg-classifier --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n sdg-classifier
```

### Issue: External IP shows <pending>

Wait a few minutes, then check again:
```bash
kubectl get service sdg-classifier-service -n sdg-classifier --watch
```

If still pending after 10 minutes, check Azure portal for any quota issues.

---

## 💰 Cost Estimation

### Azure Resources (Monthly)

**Development Setup:**
- AKS Cluster: ~$70 (2 × Standard_D2s_v3 nodes)
- Container Registry: ~$20 (Standard SKU)
- Load Balancer: ~$20
- **Total**: ~$110/month

**Production Setup:**
- AKS Cluster: ~$200 (3 nodes with auto-scaling)
- Container Registry: ~$75 (Premium SKU)
- Load Balancer: ~$20
- Monitoring: ~$50
- **Total**: ~$345/month

**Cost Saving Tips:**
- Stop AKS when not in use: `az aks stop`
- Use Basic ACR for development
- Enable cluster autoscaler (scale to 0 when idle)
- Use Azure Calculator for precise estimates

---

## 📚 Additional Resources

- **[AZURE_SETUP.md](AZURE_SETUP.md)** - Detailed Azure setup guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment documentation
- **[k8s/README.md](k8s/README.md)** - Kubernetes configuration reference
- **[README.md](README.md)** - Project overview and development guide

---

## 🆘 Getting Help

- **Issues**: Open an issue on GitHub
- **Azure Support**: Azure Portal → Support
- **Documentation**: Check the docs folder
- **Community**: Stack Overflow with tags `azure-aks`, `kubernetes`, `docker`

---

## ✅ Recommended Workflow

For **Development**:
1. Local development with `streamlit run app.py`
2. Test in Docker locally
3. Push to GitHub (triggers CI/CD)

For **Production**:
1. Setup Azure resources (one-time)
2. Configure GitHub Actions
3. Push to main branch → Auto-deploy to AKS
4. Monitor in Azure Portal

---

## 🎉 Success Checklist

- [ ] Application runs locally
- [ ] Docker image builds successfully
- [ ] Azure resources created
- [ ] Image pushed to ACR
- [ ] Deployed to AKS
- [ ] External IP accessible
- [ ] Health checks passing
- [ ] Auto-scaling configured
- [ ] Monitoring enabled
- [ ] GitHub Actions configured

---

**Ready to deploy? Start with the deployment option that best fits your needs!** 🚀
