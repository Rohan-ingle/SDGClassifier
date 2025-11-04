#!/bin/bash

# SDG Classifier - Azure AKS Deployment Script
# This script automates the deployment process to Azure Kubernetes Service

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-sdg-classifier-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-sdgclassifieracr}"
AKS_NAME="${AKS_NAME:-sdg-classifier-aks}"
IMAGE_NAME="sdg-classifier"
NAMESPACE="sdg-classifier"

# Check prerequisites
check_prerequisites() {
    print_message "Checking prerequisites..."
    
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    print_message "All prerequisites satisfied!"
}

# Azure login
azure_login() {
    print_message "Checking Azure login status..."
    if ! az account show &> /dev/null; then
        print_warning "Not logged in to Azure. Please login..."
        az login
    else
        print_message "Already logged in to Azure"
        az account show --output table
    fi
}

# Create resource group
create_resource_group() {
    print_message "Creating resource group: $RESOURCE_GROUP..."
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --output table
}

# Create Azure Container Registry
create_acr() {
    print_message "Creating Azure Container Registry: $ACR_NAME..."
    
    if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        print_warning "ACR $ACR_NAME already exists. Skipping creation."
    else
        az acr create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$ACR_NAME" \
            --sku Standard \
            --admin-enabled true \
            --output table
        
        print_message "ACR created successfully!"
    fi
}

# Create AKS cluster
create_aks() {
    print_message "Creating AKS cluster: $AKS_NAME..."
    print_warning "This may take 5-10 minutes..."
    
    if az aks show --name "$AKS_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        print_warning "AKS cluster $AKS_NAME already exists. Skipping creation."
    else
        az aks create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$AKS_NAME" \
            --node-count 2 \
            --node-vm-size Standard_D2s_v3 \
            --enable-managed-identity \
            --attach-acr "$ACR_NAME" \
            --generate-ssh-keys \
            --output table
        
        print_message "AKS cluster created successfully!"
    fi
}

# Get AKS credentials
get_aks_credentials() {
    print_message "Getting AKS credentials..."
    az aks get-credentials \
        --resource-group "$RESOURCE_GROUP" \
        --name "$AKS_NAME" \
        --overwrite-existing
    
    print_message "Credentials configured. Testing connection..."
    kubectl cluster-info
}

# Build and push Docker image
build_and_push_image() {
    print_message "Building and pushing Docker image..."
    
    # Login to ACR
    print_message "Logging in to ACR..."
    az acr login --name "$ACR_NAME"
    
    # Get commit hash for versioning
    if git rev-parse --short HEAD &> /dev/null; then
        IMAGE_TAG=$(git rev-parse --short HEAD)
    else
        IMAGE_TAG="latest"
    fi
    
    print_message "Building image with tag: $IMAGE_TAG"
    
    # Build and push using ACR build (recommended)
    az acr build \
        --registry "$ACR_NAME" \
        --image "$IMAGE_NAME:$IMAGE_TAG" \
        --image "$IMAGE_NAME:latest" \
        --file Dockerfile \
        . || {
            print_error "Failed to build image. Trying local build..."
            docker build -t "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" .
            docker push "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"
        }
    
    print_message "Image pushed successfully!"
    export IMAGE_TAG
}

# Deploy to Kubernetes
deploy_to_k8s() {
    print_message "Deploying to Kubernetes..."
    
    # Create namespace
    print_message "Creating namespace..."
    kubectl apply -f k8s/namespace.yaml
    
    # Apply ConfigMap
    print_message "Applying ConfigMap..."
    kubectl apply -f k8s/configmap.yaml
    
    # Deploy application
    print_message "Deploying application..."
    export ACR_NAME
    export IMAGE_TAG=${IMAGE_TAG:-latest}
    envsubst < k8s/deployment.yaml | kubectl apply -f - -n "$NAMESPACE"
    
    # Apply HPA
    print_message "Applying Horizontal Pod Autoscaler..."
    kubectl apply -f k8s/hpa.yaml
    
    # Wait for deployment
    print_message "Waiting for deployment to complete..."
    kubectl rollout status deployment/sdg-classifier-deployment -n "$NAMESPACE" --timeout=5m
}

# Get service details
get_service_details() {
    print_message "Getting service details..."
    
    echo ""
    echo "======================================"
    echo "Deployment Summary"
    echo "======================================"
    echo "Namespace: $NAMESPACE"
    echo "Resource Group: $RESOURCE_GROUP"
    echo "ACR: $ACR_NAME.azurecr.io"
    echo "AKS Cluster: $AKS_NAME"
    echo ""
    
    print_message "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    print_message "Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    print_message "HPA Status:"
    kubectl get hpa -n "$NAMESPACE"
    
    echo ""
    print_message "Waiting for external IP assignment..."
    for i in {1..30}; do
        EXTERNAL_IP=$(kubectl get service sdg-classifier-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -n "$EXTERNAL_IP" ]; then
            echo ""
            echo "======================================"
            echo -e "${GREEN}✓ Deployment Successful!${NC}"
            echo "======================================"
            echo -e "Access your application at: ${GREEN}http://$EXTERNAL_IP${NC}"
            echo "======================================"
            return
        fi
        echo -n "."
        sleep 10
    done
    
    print_warning "External IP not assigned yet. Check manually with:"
    echo "kubectl get service sdg-classifier-service -n $NAMESPACE"
}

# Main menu
show_menu() {
    echo ""
    echo "======================================"
    echo "SDG Classifier - AKS Deployment Menu"
    echo "======================================"
    echo "1. Full Setup (Create everything)"
    echo "2. Create Azure Resources Only"
    echo "3. Build and Deploy Application Only"
    echo "4. Check Deployment Status"
    echo "5. View Logs"
    echo "6. Delete All Resources"
    echo "7. Exit"
    echo "======================================"
    read -p "Select option (1-7): " choice
    
    case $choice in
        1) full_setup ;;
        2) setup_azure_resources ;;
        3) build_and_deploy ;;
        4) check_status ;;
        5) view_logs ;;
        6) cleanup ;;
        7) exit 0 ;;
        *) print_error "Invalid option"; show_menu ;;
    esac
}

# Full setup
full_setup() {
    check_prerequisites
    azure_login
    create_resource_group
    create_acr
    create_aks
    get_aks_credentials
    build_and_push_image
    deploy_to_k8s
    get_service_details
}

# Setup Azure resources only
setup_azure_resources() {
    check_prerequisites
    azure_login
    create_resource_group
    create_acr
    create_aks
    get_aks_credentials
    print_message "Azure resources setup complete!"
}

# Build and deploy only
build_and_deploy() {
    check_prerequisites
    azure_login
    get_aks_credentials
    build_and_push_image
    deploy_to_k8s
    get_service_details
}

# Check deployment status
check_status() {
    print_message "Checking deployment status..."
    
    echo ""
    print_message "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    print_message "Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    print_message "HPA:"
    kubectl get hpa -n "$NAMESPACE"
    
    echo ""
    print_message "Deployment:"
    kubectl describe deployment sdg-classifier-deployment -n "$NAMESPACE"
}

# View logs
view_logs() {
    print_message "Fetching logs from all pods..."
    kubectl logs -l app=sdg-classifier -n "$NAMESPACE" --tail=100 --all-containers=true
}

# Cleanup
cleanup() {
    print_warning "This will delete all resources in resource group: $RESOURCE_GROUP"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_message "Deleting resource group..."
        az group delete --name "$RESOURCE_GROUP" --yes --no-wait
        print_message "Deletion initiated. This will take a few minutes to complete."
    else
        print_message "Cleanup cancelled."
    fi
}

# Run main menu if no arguments, otherwise run full setup
if [ $# -eq 0 ]; then
    show_menu
else
    case $1 in
        --full) full_setup ;;
        --setup) setup_azure_resources ;;
        --deploy) build_and_deploy ;;
        --status) check_status ;;
        --logs) view_logs ;;
        --cleanup) cleanup ;;
        --help)
            echo "Usage: $0 [option]"
            echo "Options:"
            echo "  --full      Full setup and deployment"
            echo "  --setup     Setup Azure resources only"
            echo "  --deploy    Build and deploy application"
            echo "  --status    Check deployment status"
            echo "  --logs      View application logs"
            echo "  --cleanup   Delete all resources"
            echo "  --help      Show this help message"
            ;;
        *) print_error "Invalid option. Use --help for usage."; exit 1 ;;
    esac
fi
