# GitHub Actions Secrets Configuration Guide

This guide explains how to set up GitHub Actions secrets for automated AKS deployment.

## 📋 Required Secrets

You need to configure 4 secrets in your GitHub repository:

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AZURE_CREDENTIALS` | Service Principal JSON | `{...}` (full JSON) |
| `ACR_NAME` | Azure Container Registry name | `sdgclassifieracr` |
| `AKS_CLUSTER_NAME` | Azure Kubernetes Service cluster name | `sdg-classifier-aks` |
| `AKS_RESOURCE_GROUP` | Azure Resource Group name | `sdg-classifier-rg` |

## 🔧 Step-by-Step Setup

### Step 1: Login to Azure

```bash
az login
az account list --output table
az account set --subscription "<your-subscription-id>"
```

### Step 2: Get Your Subscription ID

```bash
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"
```

Save this for later use.

### Step 3: Create Resource Group (if not exists)

```bash
RESOURCE_GROUP="sdg-classifier-rg"
LOCATION="eastus"

az group create --name $RESOURCE_GROUP --location $LOCATION
```

### Step 4: Create Service Principal

This creates a service principal with contributor role scoped to your resource group:

```bash
az ad sp create-for-rbac \
  --name "sdg-classifier-github-sp" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth
```

**Output Example:**
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

⚠️ **IMPORTANT**: Copy the entire JSON output. You'll need this for `AZURE_CREDENTIALS`.

### Step 5: Verify Service Principal

```bash
# Get the clientId from the JSON output above
SP_CLIENT_ID="<your-client-id>"

# Verify the service principal exists
az ad sp show --id $SP_CLIENT_ID
```

### Step 6: Create Azure Resources

```bash
ACR_NAME="sdgclassifieracr"
AKS_NAME="sdg-classifier-aks"

# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Standard

# Create AKS Cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_NAME \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --generate-ssh-keys
```

### Step 7: Grant Additional Permissions

Grant the service principal access to AKS and ACR:

```bash
# Get resource IDs
ACR_ID=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query id -o tsv)
AKS_ID=$(az aks show --name $AKS_NAME --resource-group $RESOURCE_GROUP --query id -o tsv)

# Assign ACR push/pull role
az role assignment create \
  --assignee $SP_CLIENT_ID \
  --role "AcrPush" \
  --scope $ACR_ID

az role assignment create \
  --assignee $SP_CLIENT_ID \
  --role "AcrPull" \
  --scope $ACR_ID

# Assign AKS admin role
az role assignment create \
  --assignee $SP_CLIENT_ID \
  --role "Azure Kubernetes Service Cluster Admin Role" \
  --scope $AKS_ID
```

### Step 8: Collect Secret Values

You should now have these values:

1. **AZURE_CREDENTIALS**: The entire JSON from Step 4
2. **ACR_NAME**: `sdgclassifieracr` (or your custom name)
3. **AKS_CLUSTER_NAME**: `sdg-classifier-aks` (or your custom name)
4. **AKS_RESOURCE_GROUP**: `sdg-classifier-rg` (or your custom name)

## 🔐 Adding Secrets to GitHub

### Option A: GitHub Web Interface

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret:

#### Secret 1: AZURE_CREDENTIALS
- **Name**: `AZURE_CREDENTIALS`
- **Value**: Paste the entire JSON from Step 4
- Click **Add secret**

#### Secret 2: ACR_NAME
- **Name**: `ACR_NAME`
- **Value**: `sdgclassifieracr`
- Click **Add secret**

#### Secret 3: AKS_CLUSTER_NAME
- **Name**: `AKS_CLUSTER_NAME`
- **Value**: `sdg-classifier-aks`
- Click **Add secret**

#### Secret 4: AKS_RESOURCE_GROUP
- **Name**: `AKS_RESOURCE_GROUP`
- **Value**: `sdg-classifier-rg`
- Click **Add secret**

### Option B: GitHub CLI

If you have GitHub CLI installed:

```bash
# Set secrets using gh CLI
gh secret set AZURE_CREDENTIALS < azure_credentials.json
gh secret set ACR_NAME -b "sdgclassifieracr"
gh secret set AKS_CLUSTER_NAME -b "sdg-classifier-aks"
gh secret set AKS_RESOURCE_GROUP -b "sdg-classifier-rg"
```

## ✅ Verify Configuration

### 1. Check Secrets in GitHub

Go to: `Settings` → `Secrets and variables` → `Actions`

You should see 4 secrets listed:
- ✓ AZURE_CREDENTIALS
- ✓ ACR_NAME
- ✓ AKS_CLUSTER_NAME
- ✓ AKS_RESOURCE_GROUP

### 2. Test Service Principal

```bash
# Login using service principal
az login --service-principal \
  -u $SP_CLIENT_ID \
  -p "<client-secret-from-json>" \
  --tenant "<tenant-id-from-json>"

# Verify access to resource group
az group show --name $RESOURCE_GROUP

# Verify access to ACR
az acr show --name $ACR_NAME

# Verify access to AKS
az aks show --name $AKS_NAME --resource-group $RESOURCE_GROUP

# Logout
az logout
```

### 3. Trigger Workflow

```bash
# Push to main branch to trigger workflow
git add .
git commit -m "Configure GitHub Actions"
git push origin main
```

Go to: `https://github.com/<username>/SDGClassifier/actions`

Watch the workflow run. It should:
1. ✅ Build and test
2. ✅ Build Docker image
3. ✅ Push to ACR
4. ✅ Deploy to AKS
5. ✅ Health checks

## 🔄 Updating Secrets

If you need to update secrets (e.g., credentials expired):

### Regenerate Service Principal Password

```bash
az ad sp credential reset --id $SP_CLIENT_ID --create-cert
```

Update the `AZURE_CREDENTIALS` secret with new JSON.

### Rotate ACR Admin Password

```bash
az acr credential renew --name $ACR_NAME --password-name password
```

## 🛡️ Security Best Practices

1. **Never commit secrets to Git**
   - Add to `.gitignore`
   - Use environment variables locally
   
2. **Use least privilege**
   - Scope service principal to resource group only
   - Don't use subscription-level permissions unless needed

3. **Rotate credentials regularly**
   - Set expiration on service principals
   - Rotate every 90 days

4. **Audit access**
   - Monitor service principal usage in Azure Portal
   - Review GitHub Actions logs

5. **Use managed identities when possible**
   - For resources within Azure, prefer managed identities
   - Service principals only for external access (like GitHub)

## 🐛 Troubleshooting

### Error: "No subscriptions found"

```bash
# Check current account
az account show

# List all subscriptions
az account list --output table

# Set correct subscription
az account set --subscription "<subscription-id>"
```

### Error: "Insufficient privileges"

```bash
# Check role assignments
az role assignment list --assignee $SP_CLIENT_ID --output table

# Add missing roles
az role assignment create \
  --assignee $SP_CLIENT_ID \
  --role "Contributor" \
  --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP
```

### Error: "Secret not found" in GitHub Actions

1. Verify secret names match exactly (case-sensitive)
2. Check secret values don't have extra spaces
3. Ensure you're in the correct repository
4. Try removing and re-adding the secret

### Error: "Authentication failed" in workflow

1. Verify JSON format of `AZURE_CREDENTIALS`
2. Check service principal hasn't expired
3. Regenerate service principal credentials
4. Update GitHub secret

### Error: "Cannot connect to AKS cluster"

```bash
# Get AKS credentials locally
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_NAME

# Test connection
kubectl cluster-info

# If works locally, issue is with service principal permissions
```

## 📚 Additional Resources

- [Azure Service Principals](https://docs.microsoft.com/azure/active-directory/develop/app-objects-and-service-principals)
- [GitHub Actions Secrets](https://docs.github.com/actions/security-guides/encrypted-secrets)
- [Azure RBAC](https://docs.microsoft.com/azure/role-based-access-control/)
- [GitHub Actions for Azure](https://github.com/Azure/actions)

## 📝 Quick Reference

```bash
# Save these values
SUBSCRIPTION_ID="<your-subscription-id>"
RESOURCE_GROUP="sdg-classifier-rg"
ACR_NAME="sdgclassifieracr"
AKS_NAME="sdg-classifier-aks"
LOCATION="eastus"
SP_NAME="sdg-classifier-github-sp"

# Create everything
az group create --name $RESOURCE_GROUP --location $LOCATION
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Standard
az aks create --resource-group $RESOURCE_GROUP --name $AKS_NAME --node-count 2 --attach-acr $ACR_NAME
az ad sp create-for-rbac --name $SP_NAME --role contributor --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP --sdk-auth
```

---

**Need help?** Open an issue on GitHub or check the [Azure CLI documentation](https://docs.microsoft.com/cli/azure/).
