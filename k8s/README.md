# Kubernetes Deployment for SDG Classifier

This directory contains Kubernetes manifests for deploying the SDG Classifier application to Azure Kubernetes Service (AKS).

## 📁 Files Overview

- **`deployment.yaml`**: Main deployment configuration with 2 replicas, resource limits, and health checks
- **`namespace.yaml`**: Creates a dedicated namespace for the application
- **`configmap.yaml`**: Application configuration (environment variables, settings)
- **`hpa.yaml`**: Horizontal Pod Autoscaler for automatic scaling (2-10 pods)
- **`ingress.yaml`**: Ingress configuration for external access with SSL/TLS

## 🚀 Prerequisites

Before deploying, ensure you have:

1. **Azure CLI** installed and configured
2. **kubectl** installed
3. **An AKS cluster** provisioned
4. **Azure Container Registry (ACR)** set up
5. **Docker image** built and pushed to ACR

### Install Azure CLI
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Install kubectl
```bash
az aks install-cli
```

## 🔧 Setup Instructions

### 1. Create Azure Resources

#### Create Resource Group
```bash
az group create --name sdg-classifier-rg --location eastus
```

#### Create Azure Container Registry (ACR)
```bash
az acr create --resource-group sdg-classifier-rg \
  --name sdgclassifieracr \
  --sku Basic
```

#### Create AKS Cluster
```bash
az aks create \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity \
  --attach-acr sdgclassifieracr \
  --generate-ssh-keys
```

#### Get AKS Credentials
```bash
az aks get-credentials \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks
```

### 2. Build and Push Docker Image

```bash
# Login to ACR
az acr login --name sdgclassifieracr

# Build the image
docker build -t sdgclassifieracr.azurecr.io/sdg-classifier:latest .

# Push to ACR
docker push sdgclassifieracr.azurecr.io/sdg-classifier:latest
```

### 3. Deploy to AKS

#### Create Namespace
```bash
kubectl apply -f k8s/namespace.yaml
```

#### Deploy ConfigMap
```bash
kubectl apply -f k8s/configmap.yaml
```

#### Deploy Application
```bash
# Update the deployment.yaml with your ACR name and image tag
export ACR_NAME=sdgclassifieracr
export IMAGE_TAG=latest
envsubst < k8s/deployment.yaml | kubectl apply -f - -n sdg-classifier
```

#### Deploy HPA (Auto-scaling)
```bash
kubectl apply -f k8s/hpa.yaml
```

#### Optional: Deploy Ingress (for custom domain)
```bash
kubectl apply -f k8s/ingress.yaml
```

## 📊 Monitoring and Management

### Check Deployment Status
```bash
kubectl get deployments -n sdg-classifier
kubectl get pods -n sdg-classifier
kubectl get services -n sdg-classifier
```

### View Logs
```bash
kubectl logs -l app=sdg-classifier -n sdg-classifier --tail=100
```

### Check HPA Status
```bash
kubectl get hpa -n sdg-classifier
kubectl describe hpa sdg-classifier-hpa -n sdg-classifier
```

### Get External IP
```bash
kubectl get service sdg-classifier-service -n sdg-classifier
```

Wait for the `EXTERNAL-IP` to be assigned, then access your application at:
```
http://<EXTERNAL-IP>
```

## 🔄 Updating the Deployment

### Update with New Image
```bash
kubectl set image deployment/sdg-classifier-deployment \
  sdg-classifier=sdgclassifieracr.azurecr.io/sdg-classifier:v2.0 \
  -n sdg-classifier
```

### Rollout Status
```bash
kubectl rollout status deployment/sdg-classifier-deployment -n sdg-classifier
```

### Rollback to Previous Version
```bash
kubectl rollout undo deployment/sdg-classifier-deployment -n sdg-classifier
```

## 🧪 Testing the Deployment

### Port Forward for Local Testing
```bash
kubectl port-forward service/sdg-classifier-service 8501:80 -n sdg-classifier
```

Then access at: `http://localhost:8501`

### Execute Commands in Pod
```bash
POD_NAME=$(kubectl get pods -n sdg-classifier -l app=sdg-classifier -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $POD_NAME -n sdg-classifier -- /bin/bash
```

## 🔐 GitHub Actions Secrets

Configure these secrets in your GitHub repository:

- `AZURE_CREDENTIALS`: Azure service principal credentials (JSON)
- `ACR_NAME`: Azure Container Registry name
- `AKS_CLUSTER_NAME`: AKS cluster name
- `AKS_RESOURCE_GROUP`: Azure resource group name

### Create Azure Service Principal
```bash
az ad sp create-for-rbac --name "sdg-classifier-sp" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/sdg-classifier-rg \
  --sdk-auth
```

Copy the JSON output and add it as `AZURE_CREDENTIALS` secret in GitHub.

## 🛡️ Security Best Practices

1. **Use Azure Key Vault** for sensitive data
2. **Enable Pod Security Policies**
3. **Implement Network Policies**
4. **Use Private Container Registry**
5. **Enable Azure Defender for Kubernetes**
6. **Regular security scanning** with Azure Security Center

## 📈 Scaling

### Manual Scaling
```bash
kubectl scale deployment sdg-classifier-deployment --replicas=5 -n sdg-classifier
```

### Cluster Autoscaling
```bash
az aks update \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5
```

## 🧹 Cleanup

### Delete Application
```bash
kubectl delete namespace sdg-classifier
```

### Delete AKS Cluster
```bash
az aks delete --resource-group sdg-classifier-rg --name sdg-classifier-aks --yes --no-wait
```

### Delete Resource Group
```bash
az group delete --name sdg-classifier-rg --yes --no-wait
```

## 📚 Additional Resources

- [Azure Kubernetes Service Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Azure Container Registry Documentation](https://docs.microsoft.com/en-us/azure/container-registry/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

## 🆘 Troubleshooting

### Pods Not Starting
```bash
kubectl describe pod <pod-name> -n sdg-classifier
kubectl logs <pod-name> -n sdg-classifier
```

### Service Not Accessible
```bash
kubectl get events -n sdg-classifier
kubectl describe service sdg-classifier-service -n sdg-classifier
```

### Image Pull Errors
```bash
# Verify ACR integration
az aks check-acr --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --acr sdgclassifieracr.azurecr.io
```
