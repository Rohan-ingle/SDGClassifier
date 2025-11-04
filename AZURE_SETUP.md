# Azure Setup Guide for SDG Classifier

This guide walks you through setting up Azure resources for deploying the SDG Classifier application.

## 📋 Prerequisites

1. **Azure Account** with active subscription
2. **Azure CLI** installed locally
3. **Sufficient permissions** to create resources
4. **Resource quotas** for:
   - Virtual Machines (for AKS nodes)
   - Public IPs
   - Load Balancers

## 🔧 Initial Setup

### 1. Install Azure CLI

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

**macOS:**
```bash
brew install azure-cli
```

**Windows:**
Download from: https://aka.ms/installazurecliwindows

### 2. Login to Azure

```bash
az login
```

This will open a browser window for authentication.

### 3. Set Default Subscription

```bash
# List all subscriptions
az account list --output table

# Set default subscription
az account set --subscription "<subscription-id>"

# Verify
az account show
```

## 🏗️ Resource Creation

### Option 1: Using the Automated Script (Recommended)

```bash
# Make script executable
chmod +x deploy-aks.sh

# Run interactive menu
./deploy-aks.sh

# Or run directly
./deploy-aks.sh --full
```

### Option 2: Manual Setup

#### Step 1: Create Resource Group

```bash
az group create \
  --name sdg-classifier-rg \
  --location eastus
```

**Available Locations:**
- `eastus`, `eastus2`, `westus`, `westus2`
- `centralus`, `northcentralus`, `southcentralus`
- `westeurope`, `northeurope`
- `southeastasia`, `eastasia`

#### Step 2: Create Azure Container Registry

```bash
az acr create \
  --resource-group sdg-classifier-rg \
  --name sdgclassifieracr \
  --sku Standard \
  --admin-enabled true
```

**SKU Options:**
- `Basic`: Development/testing ($5/month)
- `Standard`: Production workloads ($20/month)
- `Premium`: High performance ($75/month)

**Get ACR Login Server:**
```bash
az acr show --name sdgclassifieracr --query loginServer --output tsv
```

#### Step 3: Create AKS Cluster

**Basic Configuration:**
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

**Production Configuration:**
```bash
az aks create \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --min-count 2 \
  --max-count 5 \
  --enable-cluster-autoscaler \
  --enable-managed-identity \
  --attach-acr sdgclassifieracr \
  --network-plugin azure \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --zones 1 2 3
```

**VM Size Options:**

| Size | vCPUs | RAM | Cost/month* |
|------|-------|-----|-------------|
| Standard_B2s | 2 | 4GB | ~$30 |
| Standard_D2s_v3 | 2 | 8GB | ~$70 |
| Standard_D4s_v3 | 4 | 16GB | ~$140 |
| Standard_D8s_v3 | 8 | 32GB | ~$280 |

*Approximate costs, varies by region

#### Step 4: Configure kubectl

```bash
az aks get-credentials \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --overwrite-existing

# Verify connection
kubectl cluster-info
kubectl get nodes
```

#### Step 5: Verify ACR Integration

```bash
az aks check-acr \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --acr sdgclassifieracr.azurecr.io
```

If not integrated, attach manually:
```bash
az aks update \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --attach-acr sdgclassifieracr
```

## 🔐 Security Configuration

### 1. Create Service Principal for GitHub Actions

```bash
# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Create service principal
az ad sp create-for-rbac \
  --name "sdg-classifier-github-sp" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/sdg-classifier-rg \
  --sdk-auth
```

**Output format:**
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  ...
}
```

**Save this entire JSON** as `AZURE_CREDENTIALS` secret in GitHub.

### 2. Configure RBAC for AKS

```bash
# Get AKS resource ID
AKS_ID=$(az aks show -g sdg-classifier-rg -n sdg-classifier-aks --query id -o tsv)

# Assign AKS admin role to service principal
az role assignment create \
  --assignee <SERVICE_PRINCIPAL_CLIENT_ID> \
  --role "Azure Kubernetes Service Cluster Admin Role" \
  --scope $AKS_ID
```

### 3. Enable Azure AD Integration (Optional)

```bash
az aks update \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --enable-aad \
  --aad-admin-group-object-ids <AAD_GROUP_OBJECT_ID>
```

## 📊 Enable Monitoring

### 1. Container Insights

```bash
az aks enable-addons \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --addons monitoring
```

### 2. Create Log Analytics Workspace

```bash
az monitor log-analytics workspace create \
  --resource-group sdg-classifier-rg \
  --workspace-name sdg-classifier-logs \
  --location eastus
```

### 3. Link AKS to Log Analytics

```bash
LOG_ANALYTICS_ID=$(az monitor log-analytics workspace show \
  --resource-group sdg-classifier-rg \
  --workspace-name sdg-classifier-logs \
  --query id -o tsv)

az aks enable-addons \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --addons monitoring \
  --workspace-resource-id $LOG_ANALYTICS_ID
```

## 💰 Cost Management

### 1. Enable Cost Analysis

```bash
# Create budget alert
az consumption budget create \
  --resource-group sdg-classifier-rg \
  --budget-name sdg-monthly-budget \
  --amount 100 \
  --time-period Monthly
```

### 2. Set Resource Tags

```bash
az group update \
  --name sdg-classifier-rg \
  --tags Environment=Production Project=SDGClassifier Owner=YourName
```

### 3. Stop/Start AKS (Save costs when not in use)

```bash
# Stop AKS cluster (stops billing for VMs)
az aks stop \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks

# Start AKS cluster
az aks start \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks
```

## 🔍 Troubleshooting

### Issue: Insufficient Quota

**Error:**
```
Operation could not be completed as it results in exceeding approved standardDSv3Family Cores quota.
```

**Solution:**
```bash
# Request quota increase
az vm list-usage --location eastus --output table

# Or use smaller VM size
--node-vm-size Standard_B2s
```

### Issue: ACR Authentication Failed

**Solution:**
```bash
# Re-attach ACR to AKS
az aks update \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --attach-acr sdgclassifieracr

# Verify
az aks check-acr \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --acr sdgclassifieracr.azurecr.io
```

### Issue: Cannot Connect to Cluster

**Solution:**
```bash
# Get credentials again
az aks get-credentials \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --overwrite-existing

# Check cluster status
az aks show \
  --resource-group sdg-classifier-rg \
  --name sdg-classifier-aks \
  --query powerState
```

## 🧪 Verify Setup

```bash
# Check resource group
az group show --name sdg-classifier-rg

# Check ACR
az acr show --name sdgclassifieracr

# Check AKS
az aks show --resource-group sdg-classifier-rg --name sdg-classifier-aks

# Check kubectl connection
kubectl cluster-info
kubectl get nodes
```

## 📚 Additional Resources

- [Azure AKS Pricing](https://azure.microsoft.com/pricing/details/kubernetes-service/)
- [Azure Container Registry Pricing](https://azure.microsoft.com/pricing/details/container-registry/)
- [Azure VM Pricing](https://azure.microsoft.com/pricing/details/virtual-machines/)
- [Azure Cost Calculator](https://azure.microsoft.com/pricing/calculator/)

## 🆘 Getting Help

- **Azure Support**: [Azure Portal](https://portal.azure.com) → Support
- **Documentation**: [Azure AKS Docs](https://docs.microsoft.com/azure/aks/)
- **Community**: [Azure Forums](https://docs.microsoft.com/answers/products/azure)
- **Stack Overflow**: Tag `azure-aks`

## 🔄 Next Steps

After completing this setup:

1. Configure GitHub Secrets (see DEPLOYMENT.md)
2. Push code to trigger CI/CD pipeline
3. Monitor deployment in GitHub Actions
4. Access application via external IP

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)
