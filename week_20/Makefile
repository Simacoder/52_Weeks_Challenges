########################################################################
# Kubernetes cluster operations (MLOps)
########################################################################
# create the local Kubernetes cluster
export CLUSTER_NAME=cluster-789

cluster:
	kind create cluster --config kind.yaml --name $(CLUSTER_NAME)
	kubectl config use-context kind-$(CLUSTER_NAME)
	
	@echo "Listing the nodes in the cluster"
	kubectl get nodes

# delete the local Kubernetes cluster
delete-cluster:
	kind delete cluster --name $(CLUSTER_NAME)

# list Docker images registered in the local cluster
list-images:
	docker exec -it $(CLUSTER_NAME)-control-plane crictl images

########################################################################
# ML engineer operations (ML)
########################################################################
export PORT=5005

# run with hot reloading for development
dev:
	uv run fastapi dev api.py --port $(PORT)

# build the Docker image for our api
build:
	docker build -t simple-api:v1.0.0 .

# run the Docker container for our api as a standalone docker in your machine
run:
	docker run -it -p $(PORT):5000 simple-api:v1.0.0

# push the Docker image to the local Kubernetes image registry
push:
	kind load docker-image simple-api:v1.0.0 --name $(CLUSTER_NAME)

# deploy the Docker image to the local Kubernetes cluster
deploy: build push
	kubectl apply -f manifests/deployment.yaml
	kubectl apply -f manifests/service.yaml
	kubectl wait --for=condition=ready pod -l app=simple-api --timeout=60s
	kubectl port-forward svc/simple-api $(PORT):5000

# ping the api to check if it's running
test:
	curl http://localhost:$(PORT)/health