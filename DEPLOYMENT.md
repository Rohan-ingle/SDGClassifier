# 🚀 SDG Classifier - AKS Deployment Guide

Complete guide for deploying the SDG Classifier application to Azure Kubernetes Service (AKS) using Docker containers.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
- **Azure CLI** (v2.30.0 or higher)
- **Docker** (v20.10 or higher)
- **kubectl** (v1.24 or higher)
- **Git**
- **Azure Subscription** with appropriate permissions

### Azure Resources Required
- Azure Container Registry (ACR)
- Azure Kubernetes Service (AKS)
- Azure Resource Group

### Local Development
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Verify installations
az --version
docker --version
kubectl version --client
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                     │
│  (Build → Test → Build Docker → Push ACR → Deploy AKS)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Azure Container Registry (ACR)                  │
│                 sdgclassifier.azurecr.io                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Azure Kubernetes Service (AKS) Cluster              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Namespace: sdg-classifier                           │   │
│  │                                                       │   │
│  │  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │   Pod 1        │  │   Pod 2        │            │   │
│  │  │  (App + Model) │  │  (App + Model) │  ← HPA →  │   │
│  │  └────────────────┘  └────────────────┘    2-10     │   │
│  │          │                   │                       │   │
│  │          └───────┬───────────┘                       │   │
│  │                  │                                   │   │
│  │         ┌────────▼──────────┐                       │   │
│  │         │  LoadBalancer     │                       │   │
│  │         │    Service        │                       │   │
│  │         └─────────┬─────────┘                       │   │
│  └───────────────────┼─────────────────────────────────┘   │
└────────────────────────┼───────────────────────────────────┘
                         │
                         ▼
                   External Users
```

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/Rohan-ingle/SDGClassifier.git
cd SDGClassifier
```

### 2. Azure Login
```bash
az login
az account set --subscription "<your-subscription-id>"
```

### 3. Create Infrastructure
```bash
# Set variables
RESOURCE_GROUP="sdg-classifier-rg"
LOCATION="eastus"
ACR_NAME="sdgclassifieracr"
AKS_NAME="sdg-classifier-aks"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create ACR
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Standard

# Create AKS
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_NAME \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --generate-ssh-keys
```

### 4. Build and Deploy
```bash
# Get AKS credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_NAME

# Build and push Docker image
az acr build --registry $ACR_NAME --image sdg-classifier:v1.0 .

# Deploy to AKS
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Update deployment with your ACR name
export ACR_NAME=$ACR_NAME
export IMAGE_TAG=v1.0
envsubst < k8s/deployment.yaml | kubectl apply -f - -n sdg-classifier

# Apply autoscaling
kubectl apply -f k8s/hpa.yaml
```

### 5. Access Application
```bash
# Get external IP
kubectl get service sdg-classifier-service -n sdg-classifier

# Wait for EXTERNAL-IP to be assigned, then access at:
# http://<EXTERNAL-IP>
```

## Detailed Setup

### Step 1: Azure Container Registry Setup

#### Create and Configure ACR
```bash
# Create ACR
az acr create \
  --resource-group sdg-classifier-rg \
  --name sdgclassifieracr \
  --sku Standard \
  --admin-enabled true

# Get ACR credentials
az acr credential show --name sdgclassifieracr

# Login to ACR
az acr login --name sdgclassifieracr
```

#### Build and Push Image
```bash
# Build locally and push
docker build -t sdgclassifieracr.azurecr.io/sdg-classifier:latest .
docker push sdgclassifieracr.azurecr.io/sdg-classifier:latest

# OR use Azure ACR build (recommended)
az acr build \
  --registry sdgclassifieracr \
  --image sdg-classifier:$(git rev-parse --short HEAD) \
  --image sdg-classifier:latest \
  --file Dockerfile .
```

### Step 2: AKS Cluster Configuration

#### Create AKS with Optimal Settings
```bash
az aks create \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --min-count 1 \
  --max-count 5 \
  --enable-cluster-autoscaler \
  --enable-managed-identity \
  --attach-acr sdgclassifieracr \
  --network-plugin azure \
  --enable-addons monitoring \
  --generate-ssh-keys
```

#### Configure kubectl
```bash
az aks get-credentials \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --overwrite-existing

# Verify connection
kubectl cluster-info
kubectl get nodes
```

### Step 3: Deploy Application

#### Deploy in Order
```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# 3. Deploy application
export ACR_NAME=sdgclassifieracr
export IMAGE_TAG=latest
envsubst < k8s/deployment.yaml | kubectl apply -f - -n sdg-classifier

# 4. Enable autoscaling
kubectl apply -f k8s/hpa.yaml

# 5. (Optional) Setup ingress for custom domain
kubectl apply -f k8s/ingress.yaml
```

#### Verify Deployment
```bash
# Check all resources
kubectl get all -n sdg-classifier

# Check pods
kubectl get pods -n sdg-classifier -o wide

# Check service
kubectl get svc sdg-classifier-service -n sdg-classifier

# Check HPA
kubectl get hpa -n sdg-classifier
```

## CI/CD Pipeline

### GitHub Actions Setup

#### 1. Create Azure Service Principal
```bash
az ad sp create-for-rbac \
  --name "sdg-classifier-github-sp" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/sdg-classifier-rg \
  --sdk-auth
```

Copy the entire JSON output.

#### 2. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add the following secrets:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `AZURE_CREDENTIALS` | Service principal JSON | `{...}` |
| `ACR_NAME` | Container registry name | `sdgclassifieracr` |
| `AKS_CLUSTER_NAME` | AKS cluster name | `sdg-classifier-aks` |
| `AKS_RESOURCE_GROUP` | Resource group name | `sdg-classifier-rg` |

#### 3. Trigger Deployment

The pipeline automatically triggers on:
- Push to `main` or `master` branch
- Manual trigger via workflow_dispatch

```bash
# Push changes to trigger
git add .
git commit -m "Deploy to AKS"
git push origin main
```

### Pipeline Stages

1. **Build and Test** - Install dependencies, run tests, generate coverage
2. **Build Docker Image** - Build and push to ACR with version tags
3. **Deploy to AKS** - Update Kubernetes deployment with new image
4. **Health Check** - Verify deployment health and gather logs

## Monitoring & Maintenance

### View Logs
```bash
# Real-time logs from all pods
kubectl logs -f -l app=sdg-classifier -n sdg-classifier --all-containers=true

# Logs from specific pod
kubectl logs <pod-name> -n sdg-classifier

# Previous container logs (if crashed)
kubectl logs <pod-name> -n sdg-classifier --previous
```

### Monitor Resources
```bash
# Pod resource usage
kubectl top pods -n sdg-classifier

# Node resource usage
kubectl top nodes

# HPA status
kubectl describe hpa sdg-classifier-hpa -n sdg-classifier
```

### Azure Monitor Integration
```bash
# Enable Container Insights
az aks enable-addons \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --addons monitoring
```

Access metrics in Azure Portal:
- AKS Cluster → Insights
- Container Logs
- Performance Metrics
- Live Logs

### Update Deployment

#### Rolling Update
```bash
# Build new version
docker build -t sdgclassifieracr.azurecr.io/sdg-classifier:v2.0 .
docker push sdgclassifieracr.azurecr.io/sdg-classifier:v2.0

# Update deployment
kubectl set image deployment/sdg-classifier-deployment \
  sdg-classifier=sdgclassifieracr.azurecr.io/sdg-classifier:v2.0 \
  -n sdg-classifier

# Monitor rollout
kubectl rollout status deployment/sdg-classifier-deployment -n sdg-classifier
```

#### Rollback
```bash
# View rollout history
kubectl rollout history deployment/sdg-classifier-deployment -n sdg-classifier

# Rollback to previous version
kubectl rollout undo deployment/sdg-classifier-deployment -n sdg-classifier

# Rollback to specific revision
kubectl rollout undo deployment/sdg-classifier-deployment --to-revision=2 -n sdg-classifier
```

### Scaling

#### Manual Scaling
```bash
# Scale to specific number
kubectl scale deployment sdg-classifier-deployment --replicas=5 -n sdg-classifier

# Scale cluster nodes
az aks scale \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --node-count 3
```

#### Adjust Auto-scaling
Edit `k8s/hpa.yaml` and update:
```yaml
minReplicas: 3  # Change from 2
maxReplicas: 15 # Change from 10
```

Then apply:
```bash
kubectl apply -f k8s/hpa.yaml
```

## Troubleshooting

### Pod Issues

#### Pods Not Starting
```bash
# Describe pod
kubectl describe pod <pod-name> -n sdg-classifier

# Check events
kubectl get events -n sdg-classifier --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n sdg-classifier
```

Common issues:
- **ImagePullBackOff**: Check ACR integration and image name
- **CrashLoopBackOff**: Check application logs and dependencies
- **Pending**: Check node resources and PVC binding

#### Image Pull Errors
```bash
# Verify ACR access
az aks check-acr \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --acr sdgclassifieracr.azurecr.io

# Re-attach ACR if needed
az aks update \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --attach-acr sdgclassifieracr
```

### Service Issues

#### External IP Pending
```bash
# Check service
kubectl describe service sdg-classifier-service -n sdg-classifier

# Check load balancer events
kubectl get events -n sdg-classifier | grep LoadBalancer
```

#### Can't Access Application
```bash
# Test from within cluster
kubectl run -it --rm debug --image=alpine --restart=Never -- sh
wget -O- http://sdg-classifier-service.sdg-classifier.svc.cluster.local

# Port forward for local testing
kubectl port-forward service/sdg-classifier-service 8501:80 -n sdg-classifier
# Access at http://localhost:8501
```

### Performance Issues

#### High CPU/Memory Usage
```bash
# Check resource usage
kubectl top pods -n sdg-classifier

# Increase resources in deployment.yaml
resources:
  requests:
    memory: "1Gi"    # Increase
    cpu: "500m"      # Increase
  limits:
    memory: "4Gi"    # Increase
    cpu: "2000m"     # Increase
```

### Cluster Issues

#### Node Issues
```bash
# Check node status
kubectl get nodes
kubectl describe node <node-name>

# Restart AKS nodes
az aks stop --resource-group sdg-classifier-rg --name sdg-classifier-aks
az aks start --resource-group sdg-classifier-rg --name sdg-classifier-aks
```

## Cost Optimization

### Reduce Costs
```bash
# Use smaller node size for development
--node-vm-size Standard_B2s

# Reduce minimum nodes
--min-count 1

# Stop cluster when not in use
az aks stop --resource-group sdg-classifier-rg --name sdg-classifier-aks

# Delete when done
az group delete --name sdg-classifier-rg --yes --no-wait
```

### Cost Monitoring
- Enable Azure Cost Management
- Set budget alerts
- Use Azure Advisor recommendations

## Security Best Practices

1. **Use Private ACR** - Disable public access
2. **Enable Network Policies** - Restrict pod-to-pod communication
3. **Use Azure Key Vault** - Store secrets securely
4. **Enable RBAC** - Implement role-based access control
5. **Regular Updates** - Keep AKS and images updated
6. **Scan Images** - Use Azure Security Center
7. **Use Managed Identities** - Avoid using passwords

## Additional Resources

- [Azure AKS Documentation](https://docs.microsoft.com/azure/aks/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Azure Container Registry](https://docs.microsoft.com/azure/container-registry/)
- [GitHub Actions for Azure](https://github.com/Azure/actions)

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review Azure AKS troubleshooting guide
