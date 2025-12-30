# 🚀 Distributed Deep Learning Pipeline with Kubernetes & PyTorch

[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MinIO](https://img.shields.io/badge/MinIO-C72E49?style=flat&logo=minio&logoColor=white)](https://min.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Pipeline MLOps end-to-end pour l'entraînement distribué de ResNet-18 sur CIFAR-10 avec Kubernetes, PyTorchJob et MinIO.

---

## 📋 Table des Matières

- [Vue d'Ensemble](#-vue-densemble)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline MLOps](#-pipeline-mlops)
- [Résultats](#-résultats)
- [Monitoring](#-monitoring)
- [Déploiement](#-déploiement)
- [Troubleshooting](#-troubleshooting)
- [Contributeurs](#-contributeurs)

---

## 🎯 Vue d'Ensemble

Ce projet implémente un **pipeline MLOps complet** pour l'entraînement distribué de réseaux de neurones profonds sur Kubernetes. Il démontre les meilleures pratiques de l'industrie pour le Machine Learning en production :

- ✅ **Distributed Training** : Entraînement parallèle avec PyTorch DDP
- ✅ **Container Orchestration** : Déploiement et scaling avec Kubernetes
- ✅ **Artifact Management** : Versioning des modèles avec MinIO (S3-compatible)
- ✅ **Reproducibility** : Pipeline automatisé et reproductible
- ✅ **Production-Ready** : Architecture scalable et déployable

### 🎓 Cas d'Usage

- Entraînement de modèles de vision par ordinateur à grande échelle
- MLOps dans des environnements à ressources limitées
- Prototype de pipeline de production pour startups/PME
- Projet académique de Deep Learning distribué

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Kubernetes Cluster (Minikube)       │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────────────┐          │
│  │          MinIO               │          │
│  │    (S3-Compatible Storage)   │          │
│  │                              │          │
│  │  Buckets:                    │          │
│  │  • datasets/  (CIFAR-10)     │          │
│  │  • models/    (Checkpoints)  │          │
│  │  • metrics/   (JSON logs)    │          │
│  └──────────────────────────────┘          │
│                  ↑                          │
│                  │                          │
│  ┌───────────────┴────────────────┐        │
│  │      PyTorchJob (DDP)          │        │
│  │                                 │        │
│  │  ┌──────────┐    ┌──────────┐  │        │
│  │  │  Master  │    │  Worker  │  │        │
│  │  │  (Rank 0)│◄──►│  (Rank 1)│  │        │
│  │  │          │    │          │  │        │
│  │  │ ResNet-18│    │ ResNet-18│  │        │
│  │  │ Training │    │ Training │  │        │
│  │  └──────────┘    └──────────┘  │        │
│  │                                 │        │
│  │  Communication: TCP (gloo)      │        │
│  └─────────────────────────────────┘        │
│                                             │
└─────────────────────────────────────────────┘
```

### 🔧 Technologies Utilisées

| Composant | Technologie | Version | Rôle |
|-----------|------------|---------|------|
| **Orchestration** | Kubernetes | 1.34+ | Gestion des conteneurs |
| **Cluster** | Minikube | 1.37+ | Cluster Kubernetes local |
| **Training Framework** | PyTorch | 2.0.0 | Deep Learning |
| **Distributed Training** | PyTorch DDP | - | Parallélisation |
| **Job Operator** | Kubeflow Training Operator | 1.8.1 | Gestion PyTorchJob |
| **Storage** | MinIO | 2023-09 | Object storage (S3) |
| **Dataset** | CIFAR-10 | - | 60K images 32x32 |
| **Model** | ResNet-18 | - | CNN (11M params) |

---

## ✨ Fonctionnalités

### 🎯 Pipeline MLOps Complet

1. **Data Ingestion**
   - Téléchargement automatique de CIFAR-10
   - Sauvegarde dans MinIO pour réutilisation
   - Support de datasets personnalisés

2. **Feature Engineering**
   - Data augmentation (flip, crop)
   - Normalisation selon statistiques CIFAR-10
   - Distribution intelligente des données entre workers

3. **Distributed Training**
   - PyTorch Distributed Data Parallel (DDP)
   - Communication backend : Gloo (CPU-optimized)
   - Synchronisation automatique des gradients
   - 1 Master + 1 Worker (extensible à N workers)

4. **Model Evaluation**
   - Métriques calculées à chaque epoch
   - Train accuracy, Test accuracy
   - Train loss, Test loss

5. **Model Versioning**
   - Checkpoints sauvegardés à chaque epoch
   - Versioning avec timestamps
   - Modèle "latest" toujours disponible

6. **Automated Deployment**
   - Modèles prêts pour déploiement
   - API REST pour inférence (optionnel)

### 🚀 Optimisations

- **Efficacité mémoire** : Batch size optimisé, gradient accumulation
- **Vitesse** : DataLoader multi-threaded, pin_memory
- **Robustesse** : Gestion d'erreurs, retry automatique
- **Observabilité** : Logs détaillés, métriques structurées

---

## 📦 Prérequis

### Système d'Exploitation

- **Windows** : Windows 10/11 avec WSL2
- **Linux** : Ubuntu 20.04+ ou équivalent
- **macOS** : macOS 11+ avec Docker Desktop

### Ressources Matérielles

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **CPU** | 2 cores | 4 cores |
| **RAM** | 6 GB | 8 GB |
| **Disque** | 20 GB | 30 GB |
| **GPU** | Optionnel | Optionnel |

### Logiciels

- Docker 20.10+
- Kubernetes (via Minikube)
- Python 3.9+
- Git

---

## 🔧 Installation

### Étape 1 : Environnement de Base

#### Sur Windows (WSL2)

```powershell
# Dans PowerShell en Administrateur
wsl --install
# Redémarrer Windows
```

Puis dans WSL2 Ubuntu :

```bash
# Mise à jour système
sudo apt update && sudo apt upgrade -y

# Installation Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Fermer et rouvrir WSL pour appliquer les changements
exit
# Puis rouvrir WSL
```

#### Sur Linux

```bash
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Vérifier
docker --version
```

### Étape 2 : Kubernetes (Minikube)

```bash
# Télécharger Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Installer kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/

# Vérifier les installations
minikube version
kubectl version --client
```

### Étape 3 : Démarrage du Cluster

```bash
# Créer un cluster Kubernetes local
minikube start --cpus=2 --memory=6144 --disk-size=20g --driver=docker

# Vérifier que le cluster fonctionne
kubectl cluster-info
kubectl get nodes
```

**Sortie attendue** :
```
NAME       STATUS   ROLES           AGE   VERSION
minikube   Ready    control-plane   1m    v1.34.0
```

### Étape 4 : Kubeflow Training Operator

```bash
# Cloner le repository
git clone --depth 1 --branch v1.8.1 https://github.com/kubeflow/training-operator.git
cd training-operator

# Installer l'opérateur
kubectl apply -k manifests/overlays/standalone

# Attendre que ce soit prêt
kubectl wait --for=condition=ready pod -l app=training-operator -n kubeflow --timeout=300s

# Vérifier
kubectl get pods -n kubeflow
```

**Sortie attendue** :
```
NAME                                 READY   STATUS    RESTARTS   AGE
training-operator-xxxxx-xxxxx        1/1     Running   0          1m
```

### Étape 5 : Dépendances Python

```bash
# Créer un environnement virtuel
cd ~
mkdir mlops-distributed-project
cd mlops-distributed-project

python3 -m venv venv
source venv/bin/activate

# Installer les packages
pip install --upgrade pip
pip install torch torchvision minio requests pillow
```

---

## 🚀 Utilisation

### Cloner le Projet

```bash
cd ~/mlops-distributed-project

# Si vous avez un repository Git
git clone <votre-repo-url> .

# OU créer les fichiers manuellement (voir ci-dessous)
```

### Créer les Fichiers de Configuration

#### Fichier 1 : `lightweight-pipeline.yaml`

```bash
nano lightweight-pipeline.yaml
```

Copiez le contenu complet du pipeline (voir [lightweight-pipeline.yaml](./lightweight-pipeline.yaml)).

**Points clés du fichier** :
- Namespace `mlops-light`
- Déploiement MinIO
- ConfigMap avec le code d'entraînement
- PyTorchJob avec Master + Worker
- Ressources CPU/Mémoire optimisées

#### Fichier 2 : `quick-run.sh`

```bash
nano quick-run.sh
chmod +x quick-run.sh
```

Copiez le contenu du script de lancement (voir [quick-run.sh](./quick-run.sh)).

### Lancer le Pipeline

```bash
# Tout en une commande
./quick-run.sh
```

**Ce script va** :
1. ✅ Nettoyer les namespaces précédents
2. ✅ Déployer MinIO
3. ✅ Créer les buckets nécessaires
4. ✅ Lancer le PyTorchJob distribué
5. ✅ Afficher les logs en temps réel

### Sortie Attendue

```
==========================================
  Version ALLÉGÉE - Sans MLflow
  Uniquement: MinIO + PyTorchJob
==========================================

[1/5] Nettoyage...
namespace "mlops-light" deleted

[2/5] Déploiement de l'infrastructure...
namespace/mlops-light created
deployment.apps/minio created
service/minio created
pytorchjob.kubeflow.org/resnet-light created

[3/5] Attente de MinIO...
pod/minio-xxxxx condition met

[4/5] Configuration des buckets MinIO...
✓ Bucket 'datasets' créé
✓ Bucket 'models' créé
✓ Bucket 'metrics' créé

[5/5] Lancement de l'entraînement distribué...
Logs en temps réel:
[Rank 0/2] 🚀 DÉMARRAGE ENTRAÎNEMENT DISTRIBUÉ
...
```

**Durée totale** : ~25-35 minutes
- Setup : 2-3 min
- Entraînement : 20-30 min

---

## 📊 Pipeline MLOps

Le pipeline est composé de **5 étapes principales** :

### 1️⃣ Data Ingestion

```python
# Téléchargement automatique de CIFAR-10
trainset = torchvision.datasets.CIFAR10(
    root='/data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='/data', train=False, download=True, transform=transform_test
)
```

**Résultat** : 50,000 images d'entraînement + 10,000 images de test

### 2️⃣ Feature Engineering

```python
# Transformations et augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # Flip horizontal aléatoire
    transforms.RandomCrop(32, padding=4),    # Crop aléatoire
    transforms.ToTensor(),                   # Conversion en tenseur
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))  # Normalisation
])
```

**Résultat** : DataLoaders configurés avec DistributedSampler

### 3️⃣ Distributed Training

```python
# Configuration du training distribué
if world_size > 1:
    dist.init_process_group(backend='gloo', ...)
    model = DDP(model)

# Entraînement sur 10 epochs
for epoch in range(10):
    # Training loop avec synchronisation automatique
    ...
```

**Stratégie** : 
- PyTorch Distributed Data Parallel (DDP)
- Backend Gloo (optimisé CPU)
- Synchronisation des gradients automatique

### 4️⃣ Model Evaluation

```python
# Évaluation à chaque epoch
model.eval()
with torch.no_grad():
    for inputs, targets in testloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Calcul accuracy
        ...
```

**Métriques calculées** :
- Train Loss & Accuracy
- Test Loss & Accuracy

### 5️⃣ Model Versioning

```python
# Sauvegarde avec timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Checkpoint à chaque epoch
torch.save(checkpoint, f'/output/checkpoint_epoch_{epoch+1}.pth')

# Upload vers MinIO
minio_client.fput_object(
    "models",
    f"resnet-cifar10/model_v{timestamp}.pth",
    final_model_path
)
```

**Organisation MinIO** :
```
models/
└── resnet-cifar10/
    ├── checkpoint_epoch_1.pth
    ├── checkpoint_epoch_2.pth
    ├── ...
    ├── checkpoint_epoch_10.pth
    ├── model_v20241228_143022.pth
    └── model_latest.pth

metrics/
└── resnet-cifar10/
    └── metrics_v20241228_143022.json
```

---

## 📈 Résultats

### Métriques de Performance

**Configuration** :
- Modèle : ResNet-18 (11M paramètres)
- Dataset : CIFAR-10 (60K images)
- Epochs : 10
- Batch Size : 128
- Learning Rate : 0.1 (avec cosine annealing)
- Optimizer : SGD (momentum=0.9, weight_decay=5e-4)
- Workers : 1 Master + 1 Worker

**Résultats Attendus** :

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1 | 1.875 | 31.2% | 1.652 | 39.8% |
| 2 | 1.512 | 45.6% | 1.398 | 50.2% |
| 5 | 0.983 | 65.4% | 0.945 | 67.1% |
| 10 | 0.421 | 85.3% | 0.682 | 78.5% |

**Graphiques** :

```
Train vs Test Accuracy
90% ┤                                    ╭──
80% ┤                           ╭────────╯
70% ┤                    ╭──────╯
60% ┤              ╭─────╯
50% ┤        ╭─────╯
40% ┤   ╭────╯
30% ┼───╯
    └─────────────────────────────────────
    1   2   3   4   5   6   7   8   9   10
                    Epochs
```

### Temps d'Exécution

| Phase | Durée | Description |
|-------|-------|-------------|
| Setup | 2-3 min | Déploiement MinIO, création buckets |
| Data Download | 1-2 min | CIFAR-10 download (première fois) |
| Training | 20-25 min | 10 epochs complets |
| Saving | 1-2 min | Upload vers MinIO |
| **Total** | **25-30 min** | Pipeline complet |

### Utilisation Ressources

```bash
# Pendant l'entraînement
kubectl top nodes
kubectl top pods -n mlops-light
```

**Ressources typiques** :
- Master : ~1.5 GB RAM, 80% CPU
- Worker : ~1.5 GB RAM, 80% CPU
- MinIO : ~250 MB RAM, 10% CPU

---

## 🔍 Monitoring

### Surveiller l'Entraînement

#### 1. Status du PyTorchJob

```bash
kubectl get pytorchjob -n mlops-light
```

**Sortie** :
```
NAME           STATE       AGE
resnet-light   Running     15m
```

**États possibles** :
- `Created` : Job créé, pods en cours de démarrage
- `Running` : Entraînement en cours
- `Succeeded` : Entraînement terminé avec succès
- `Failed` : Échec de l'entraînement

#### 2. Pods d'Entraînement

```bash
kubectl get pods -n mlops-light
```

**Sortie** :
```
NAME                      READY   STATUS    RESTARTS   AGE
minio-xxxxx               1/1     Running   0          20m
resnet-light-master-0     1/1     Running   0          18m
resnet-light-worker-0     1/1     Running   0          18m
```

#### 3. Logs en Temps Réel

**Master (Rank 0)** :
```bash
kubectl logs -f -l training.kubeflow.org/job-name=resnet-light,training.kubeflow.org/replica-type=master -n mlops-light
```

**Worker (Rank 1)** :
```bash
kubectl logs -f -l training.kubeflow.org/job-name=resnet-light,training.kubeflow.org/replica-type=worker -n mlops-light
```

**Exemple de logs** :
```
============================================================
[Rank 0/2] 🚀 DÉMARRAGE ENTRAÎNEMENT DISTRIBUÉ
============================================================

[Rank 0] 📥 ÉTAPE 1/5: Data Ingestion
[Rank 0] ✓ 50000 train, 10000 test

[Rank 0] 🔧 ÉTAPE 2/5: Feature Engineering
[Rank 0] ✓ DataLoaders configurés

[Rank 0] 🏗️  ÉTAPE 3/5: Model Creation
[Rank 0] ✓ ResNet-18 créé

[Rank 0] 🎯 Entraînement 10 epochs...

  Epoch 1/10 [0/391] Loss: 2.303 Acc: 9.38%
  Epoch 1/10 [50/391] Loss: 2.156 Acc: 18.75%
  ...
```

### Accéder à MinIO Console

```bash
# Port-forwarding
kubectl port-forward -n mlops-light svc/minio 9001:9001
```

Puis ouvrir dans le navigateur : **http://localhost:9001**

**Credentials** :
- Username : `minioadmin`
- Password : `minioadmin`

**Navigation** :
1. Cliquer sur "Buckets"
2. Sélectionner `models`
3. Naviguer dans `resnet-cifar10/`
4. Télécharger les checkpoints et métriques

### Visualiser les Métriques

```bash
# Télécharger les métriques
kubectl port-forward -n mlops-light svc/minio 9000:9000 &

# Script Python pour visualiser
python3 << 'EOF'
from minio import Minio
import json
import tempfile

client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

# Liste des fichiers de métriques
objects = list(client.list_objects("metrics", prefix="resnet-cifar10/", recursive=True))

if objects:
    # Télécharger le plus récent
    latest = sorted(objects, key=lambda x: x.last_modified, reverse=True)[0]
    
    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
        client.fget_object("metrics", latest.object_name, tmp.name)
        with open(tmp.name, 'r') as f:
            metrics = json.load(f)
    
    print("\n" + "="*60)
    print("           RÉSULTATS FINAUX")
    print("="*60)
    print(f"\n📊 Epochs: {len(metrics['epochs'])}")
    print(f"\n🎯 Accuracy Finale:")
    print(f"   Train: {metrics['train_acc'][-1]:.2f}%")
    print(f"   Test:  {metrics['test_acc'][-1]:.2f}%")
    print(f"\n📈 Meilleure Accuracy:")
    print(f"   Train: {max(metrics['train_acc']):.2f}%")
    print(f"   Test:  {max(metrics['test_acc']):.2f}%")
else:
    print("❌ Aucune métrique trouvée")
EOF
```

---

## 🚀 Déploiement

### Déployer le Modèle pour Inférence

Une fois l'entraînement terminé, déployez une API REST pour faire des prédictions.

#### Créer le Service d'Inférence

```bash
nano deploy-inference.yaml
```

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-code
  namespace: mlops-light
data:
  serve.py: |
    from flask import Flask, request, jsonify
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    import io
    from minio import Minio
    
    app = Flask(__name__)
    
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    def load_model():
        client = Minio("minio.mlops-light.svc.cluster.local:9000",
                      access_key="minioadmin", secret_key="minioadmin", secure=False)
        
        client.fget_object("models", "resnet-cifar10/model_latest.pth", "/tmp/model.pth")
        
        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        
        checkpoint = torch.load("/tmp/model.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    model = load_model()
    
    @app.route('/predict', methods=['POST'])
    def predict():
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        return jsonify({
            "prediction": CLASSES[pred.item()],
            "confidence": float(conf.item())
        })
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8080)

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resnet-inference
  namespace: mlops-light
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: api
        image: python:3.9
        command:
        - bash
        - -c
        - |
          pip install flask torch torchvision minio pillow
          python /app/serve.py
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: code
          mountPath: /app
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: code
        configMap:
          name: inference-code
---
apiVersion: v1
kind: Service
metadata:
  name: inference
  namespace: mlops-light
spec:
  type: ClusterIP
  ports:
  - port: 8080
    targetPort: 8080
  selector:
    app: inference
```

#### Déployer

```bash
kubectl apply -f deploy-inference.yaml

# Attendre que le pod soit prêt
kubectl wait --for=condition=ready pod -l app=inference -n mlops-light --timeout=300s
```

#### Tester l'API

```bash
# Port-forward
kubectl port-forward -n mlops-light svc/inference 8080:8080 &

# Test avec une image
curl -X POST -F "file=@test_image.png" http://localhost:8080/predict
```

**Réponse attendue** :
```json
{
  "prediction": "cat",
  "confidence": 0.9234
}
```

---

## 🛠️ Troubleshooting

### Problèmes Courants

#### 1. Pods en `Pending`

**Symptôme** :
```bash
kubectl get pods -n mlops-light
# resnet-light-master-0   0/1   Pending   0   5m
```

**Cause** : Ressources insuffisantes

**Solution** :
```bash
# Supprimer et recréer Minikube avec plus de mémoire
minikube delete
minikube start --cpus=2 --memory=8192 --disk-size=20g --driver=docker
```

#### 2. `ImagePullBackOff`

**Symptôme** :
```bash
kubectl describe pod -n mlops-light resnet-light-master-0
# Warning  Failed  ... Failed to pull image "pytorch/pytorch:2.0.0"
```

**Cause** : Problème réseau ou image inexistante

**Solution** :
```bash
# Vérifier la connexion
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Si ça échoue, utiliser une image plus légère
# Éditer lightweight-pipeline.yaml :
# image: pytorch/pytorch:2.0.0-cpu
```

#### 3. Pods en `CrashLoopBackOff`

**Symptôme** :
```bash
kubectl get pods -n mlops-light
# resnet-light-master-0   0/1   CrashLoopBackOff   3   5m
```

**Solution** :
```bash
# Voir les logs d'erreur
kubectl logs -n mlops-light resnet-light-master-0

# Problèmes courants :
# - Erreur Python → Vérifier le code dans le ConfigMap
# - Erreur MinIO → Vérifier que MinIO est Running
# - OOMKilled → Augmenter les ressources mémoire
```

#### 4. MinIO Inaccessible

**Symptôme** :
```
⚠️  MinIO: connection refused
```

**Solution** :
```bash
# Vérifier MinIO
kubectl get pods -n mlops-light -l app=minio

# Redémarrer MinIO
kubectl rollout restart deployment minio -n mlops-light

# Attendre
kubectl wait --for=condition=ready pod -l app=minio -n mlops-light --timeout=120s
```

#### 5. Training Operator Non Installé

**Symptôme** :
```
error: unable to recognize "lightweight-pipeline.yaml": 
no matches for kind "PyTorchJob"
```

**Solution** :
```bash
# Vérifier l'installation
kubectl get pods -n kubeflow

# Si absent, réinstaller
cd ~/mlops-distributed-project/training-operator
kubectl apply -k manif