#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  Version ALLÉGÉE - Sans MLflow"
echo "  Uniquement: MinIO + PyTorchJob"
echo "=========================================="

# Nettoyer l'ancien namespace
echo -e "\n${BLUE}[1/5]${NC} Nettoyage..."
kubectl delete namespace mlops --ignore-not-found=true
kubectl delete namespace mlops-light --ignore-not-found=true
sleep 10

# Déployer
echo -e "\n${GREEN}[2/5]${NC} Déploiement de l'infrastructure..."
kubectl apply -f lightweight-pipeline.yaml

# Attendre MinIO
echo -e "\n${BLUE}[3/5]${NC} Attente de MinIO..."
kubectl wait --for=condition=ready pod -l app=minio -n mlops-light --timeout=300s

# Créer les buckets
echo -e "\n${GREEN}[4/5]${NC} Configuration des buckets MinIO..."
kubectl port-forward -n mlops-light svc/minio 9000:9000 > /dev/null 2>&1 &
PF_PID=$!
sleep 5

python3 << 'PYEOF'
from minio import Minio
try:
    client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
    for bucket in ["datasets", "models", "metrics"]:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"✓ Bucket '{bucket}' créé")
except Exception as e:
    print(f"⚠️  {e}")
PYEOF

kill $PF_PID 2>/dev/null

# Lancer l'entraînement
echo -e "\n${GREEN}[5/5]${NC} Lancement de l'entraînement distribué..."
echo ""
kubectl get pytorchjob -n mlops-light
echo ""
kubectl get pods -n mlops-light

# Attendre le démarrage
echo -e "\n${YELLOW}Attente du démarrage des pods (peut prendre 2-3 min)...${NC}"
sleep 30

kubectl get pods -n mlops-light

echo -e "\n${BLUE}Logs en temps réel:${NC}"
echo "Appuyez sur Ctrl+C pour arrêter les logs"
echo ""

kubectl wait --for=condition=ready pod -l training.kubeflow.org/job-name=resnet-light,training.kubeflow.org/replica-type=master -n mlops-light --timeout=600s

kubectl logs -f -l training.kubeflow.org/job-name=resnet-light,training.kubeflow.org/replica-type=master -n mlops-light

echo -e "\n${GREEN}=========================================="
echo "  ✅ PIPELINE TERMINÉ"
echo "==========================================${NC}"
echo ""
echo "📦 Accéder à MinIO:"
echo "   kubectl port-forward -n mlops-light svc/minio 9001:9001"
echo "   http://localhost:9001 (minioadmin / minioadmin)"
echo ""
echo "🔍 Commandes utiles:"
echo "   kubectl get pytorchjob -n mlops-light"
echo "   kubectl get pods -n mlops-light"
echo "   kubectl logs -f -l training.kubeflow.org/replica-type=worker -n mlops-light"
echo ""
