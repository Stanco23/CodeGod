# Deployment Guide

Complete guide for deploying Code-God in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Considerations](#production-considerations)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 16GB RAM minimum (32GB recommended)
- GPU optional (for faster inference)

### Setup

1. **Clone repository**:
```bash
git clone https://github.com/yourusername/code-god.git
cd code-god
```

2. **Install dependencies**:
```bash
# Master AI dependencies
pip install -r requirements/master-requirements.txt

# Worker dependencies
pip install -r requirements/worker-requirements.txt
```

3. **Start infrastructure**:
```bash
docker-compose up -d chromadb redis postgres
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run Master AI**:
```bash
export ANTHROPIC_API_KEY=your_key
python master_orchestrator.py
```

6. **Run Workers** (in separate terminals):
```bash
# Backend worker
export WORKER_TYPE=backend
python worker_agent.py

# Frontend worker
export WORKER_TYPE=frontend
python worker_agent.py

# Testing worker
export WORKER_TYPE=testing
python worker_agent.py
```

---

## Docker Deployment

### Quick Start

```bash
./scripts/start.sh
```

This automatically:
- Checks prerequisites
- Creates necessary directories
- Builds Docker images
- Starts all services
- Verifies health

### Manual Deployment

1. **Build images**:
```bash
docker-compose build
```

2. **Start services**:
```bash
docker-compose up -d
```

3. **Verify health**:
```bash
curl http://localhost:8001/health
```

4. **View logs**:
```bash
docker-compose logs -f
```

### Service Configuration

#### Scaling Workers

Increase worker replicas for better performance:

```bash
docker-compose up -d --scale worker-backend=4 --scale worker-frontend=4
```

Or modify `docker-compose.yml`:

```yaml
worker-backend:
  deploy:
    replicas: 4
```

#### Resource Limits

Configure CPU and memory limits:

```yaml
worker-backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

#### GPU Support

Enable GPU for workers:

```yaml
worker-backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Cloud Deployment

### AWS

#### Using ECS (Elastic Container Service)

1. **Push images to ECR**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag codegod-master:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/codegod-master:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/codegod-master:latest
```

2. **Create ECS task definitions** for:
   - Master AI
   - Worker agents (backend, frontend, testing, devops)
   - Infrastructure (Redis, ChromaDB)

3. **Configure services**:
   - Use Application Load Balancer for Master AI
   - Set up auto-scaling for workers
   - Use EFS for persistent storage

4. **Use managed services**:
   - RDS for PostgreSQL
   - ElastiCache for Redis
   - S3 for artifacts

#### Using EKS (Elastic Kubernetes Service)

See [Kubernetes Deployment](#kubernetes-deployment) section.

### Google Cloud Platform

#### Using Cloud Run

1. **Build and push images**:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/codegod-master
```

2. **Deploy services**:
```bash
gcloud run deploy codegod-master \
  --image gcr.io/PROJECT_ID/codegod-master \
  --platform managed \
  --memory 4Gi \
  --timeout 900 \
  --set-env-vars ANTHROPIC_API_KEY=your_key
```

3. **Use managed services**:
   - Cloud SQL for PostgreSQL
   - Memorystore for Redis
   - Cloud Storage for artifacts

### Azure

#### Using Container Instances

1. **Create resource group**:
```bash
az group create --name codegod-rg --location eastus
```

2. **Create container instances**:
```bash
az container create \
  --resource-group codegod-rg \
  --name codegod-master \
  --image codegod-master:latest \
  --cpu 4 \
  --memory 8 \
  --ports 8001 \
  --environment-variables ANTHROPIC_API_KEY=your_key
```

3. **Use managed services**:
   - Azure Database for PostgreSQL
   - Azure Cache for Redis

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3

### Deployment Steps

1. **Create namespace**:
```bash
kubectl create namespace codegod
```

2. **Create secrets**:
```bash
kubectl create secret generic codegod-secrets \
  --from-literal=anthropic-api-key=YOUR_KEY \
  --from-literal=postgres-password=SECURE_PASSWORD \
  -n codegod
```

3. **Deploy infrastructure**:

`k8s/chromadb.yaml`:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chromadb
  namespace: codegod
spec:
  serviceName: chromadb
  replicas: 1
  selector:
    matchLabels:
      app: chromadb
  template:
    metadata:
      labels:
        app: chromadb
    spec:
      containers:
      - name: chromadb
        image: chromadb/chroma:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: data
          mountPath: /chroma/chroma
        env:
        - name: IS_PERSISTENT
          value: "TRUE"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: chromadb
  namespace: codegod
spec:
  selector:
    app: chromadb
  ports:
  - port: 8000
    targetPort: 8000
```

4. **Deploy Master AI**:

`k8s/master.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: master-ai
  namespace: codegod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: master-ai
  template:
    metadata:
      labels:
        app: master-ai
    spec:
      containers:
      - name: master-ai
        image: codegod-master:latest
        ports:
        - containerPort: 8001
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: codegod-secrets
              key: anthropic-api-key
        - name: CHROMADB_HOST
          value: chromadb
        - name: REDIS_HOST
          value: redis
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: master-ai
  namespace: codegod
spec:
  type: LoadBalancer
  selector:
    app: master-ai
  ports:
  - port: 80
    targetPort: 8001
```

5. **Deploy Workers**:

`k8s/workers.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker-backend
  namespace: codegod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: worker-backend
  template:
    metadata:
      labels:
        app: worker-backend
    spec:
      containers:
      - name: worker
        image: codegod-worker:latest
        env:
        - name: WORKER_TYPE
          value: backend
        - name: WORKER_MODEL
          value: phi-3:3b
        - name: MASTER_HOST
          value: master-ai
        - name: REDIS_HOST
          value: redis
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

6. **Apply configurations**:
```bash
kubectl apply -f k8s/
```

7. **Verify deployment**:
```bash
kubectl get pods -n codegod
kubectl logs -f deployment/master-ai -n codegod
```

### Auto-scaling

Enable Horizontal Pod Autoscaler for workers:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-backend-hpa
  namespace: codegod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Production Considerations

### Security

1. **API Keys**:
   - Use secret management (AWS Secrets Manager, HashiCorp Vault)
   - Rotate keys regularly
   - Never commit to git

2. **Network Security**:
   - Use VPC/private networks
   - Configure security groups/firewall rules
   - Enable SSL/TLS for all external endpoints

3. **Container Security**:
   - Scan images for vulnerabilities
   - Run as non-root user
   - Use minimal base images
   - Keep dependencies updated

4. **Access Control**:
   - Implement authentication for API
   - Use RBAC for Kubernetes
   - Enable audit logging

### High Availability

1. **Redundancy**:
   - Run multiple Master AI instances behind load balancer
   - Deploy workers across availability zones
   - Use managed database services with replication

2. **Health Checks**:
   - Configure liveness and readiness probes
   - Implement circuit breakers
   - Set up failover mechanisms

3. **Backup & Recovery**:
   - Automated daily backups
   - Test restore procedures
   - Document recovery process

### Performance Optimization

1. **Caching**:
   - Redis for task queue and caching
   - CDN for static assets
   - Browser caching headers

2. **Database**:
   - Connection pooling
   - Query optimization
   - Read replicas for scaling

3. **Resource Allocation**:
   - Right-size containers
   - Use GPU instances for workers if budget allows
   - Monitor and adjust based on metrics

### Cost Optimization

1. **Compute**:
   - Use spot/preemptible instances for workers
   - Auto-scale based on load
   - Schedule downtime for non-production environments

2. **Storage**:
   - Use appropriate storage tiers
   - Implement data lifecycle policies
   - Compress backups

3. **API Costs**:
   - Monitor token usage
   - Implement rate limiting
   - Cache API responses when possible

---

## Monitoring & Maintenance

### Monitoring Stack

Enable monitoring with Prometheus and Grafana:

```bash
docker-compose --profile monitoring up -d
```

Access Grafana at `http://localhost:3000`

### Key Metrics

1. **System Metrics**:
   - CPU and memory usage
   - Network I/O
   - Disk usage

2. **Application Metrics**:
   - Tasks per second
   - Task completion rate
   - Error rate
   - Average execution time

3. **AI Metrics**:
   - Token consumption
   - Model inference time
   - API latency

### Logging

Centralized logging with ELK or cloud provider:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Alerting

Configure alerts for:
- High error rates
- Service downtime
- Resource exhaustion
- Failed tasks
- API quota limits

### Backup Procedures

1. **Automated backups**:
```bash
# Add to crontab
0 2 * * * /path/to/code-god/scripts/backup.sh
```

2. **Manual backup**:
```bash
./scripts/backup.sh
```

3. **Restore**:
```bash
# Restore PostgreSQL
docker-compose exec -T postgres psql -U codegod codegod < backups/postgres_TIMESTAMP.sql

# Restore ChromaDB
docker-compose exec -T chromadb tar xzf - -C /chroma/chroma < backups/chromadb_TIMESTAMP.tar.gz
```

### Maintenance Tasks

**Weekly**:
- Review logs for errors
- Check resource usage
- Verify backups

**Monthly**:
- Update dependencies
- Security patches
- Performance review
- Cost analysis

**Quarterly**:
- Disaster recovery test
- Architecture review
- Capacity planning

---

## Troubleshooting

### Common Issues

**Service won't start**:
```bash
# Check logs
docker-compose logs SERVICE_NAME

# Restart service
docker-compose restart SERVICE_NAME
```

**Out of memory**:
```bash
# Reduce worker replicas or model size
docker-compose up -d --scale worker-backend=1
```

**Task timeout**:
```bash
# Increase timeout in .env
TASK_TIMEOUT=600
docker-compose restart master-ai
```

**Database connection errors**:
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Reset database
docker-compose down postgres
docker volume rm codegod_postgres_data
docker-compose up -d postgres
```

### Support

For issues not covered here:
- Check GitHub Issues
- Review logs with LOG_LEVEL=DEBUG
- Contact support team
