#!/usr/bin/env python3
from minio import Minio
from datetime import datetime
import os

print("🔌 Connexion à MinIO...")

client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Upload checkpoint_epoch_9 comme modèle final
if os.path.exists('model_checkpoint_9.pth'):
    print(f"📤 Upload du checkpoint epoch 9 comme modèle final...")
    client.fput_object("models", f"resnet-cifar10/model_v{timestamp}.pth", "model_checkpoint_9.pth")
    client.fput_object("models", "resnet-cifar10/model_latest.pth", "model_checkpoint_9.pth")
    print(f"✅ Modèle uploadé: model_v{timestamp}.pth")
    print(f"✅ Modèle uploadé: model_latest.pth")
else:
    print("⚠️  model_checkpoint_9.pth non trouvé")
    print("   Téléchargez-le d'abord avec:")
    print("   kubectl cp mlops-light/pvc-viewer:/output/checkpoint_epoch_9.pth ./model_checkpoint_9.pth")

# Upload métriques
if os.path.exists('metrics.json'):
    print(f"📤 Upload des métriques...")
    client.fput_object("metrics", f"resnet-cifar10/metrics_v{timestamp}.json", "metrics.json")
    print(f"✅ Métriques uploadées: metrics_v{timestamp}.json")
else:
    print("❌ metrics.json non trouvé")

# Afficher le contenu
print("\n" + "="*60)
print("📦 CONTENU DE MINIO")
print("="*60)

print("\n🗂️  Bucket 'models':")
objects = list(client.list_objects("models", recursive=True))
for obj in sorted(objects, key=lambda x: x.object_name):
    size_mb = obj.size / (1024 * 1024)
    print(f"  • {obj.object_name} ({size_mb:.1f} MB)")

print("\n📊 Bucket 'metrics':")
for obj in client.list_objects("metrics", recursive=True):
    print(f"  • {obj.object_name}")

print("\n📁 Bucket 'datasets':")
for obj in client.list_objects("datasets", recursive=True):
    size_mb = obj.size / (1024 * 1024)
    print(f"  • {obj.object_name} ({size_mb:.1f} MB)")

print("\n✅ Upload terminé!")
