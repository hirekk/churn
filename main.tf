terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.6.2"
    }
  }
}

# Variables for Kaggle credentials
variable "kaggle_username" {
  description = "Kaggle username for dataset download"
  type        = string
  sensitive   = true
}

variable "kaggle_key" {
  description = "Kaggle API key for dataset download"
  type        = string
  sensitive   = true
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Create the registry container
resource "docker_container" "registry" {
  name  = "local-registry"
  image = docker_image.registry.image_id

  network_mode = "host"

  env = [
    "REGISTRY_STORAGE_DELETE_ENABLED=true",
    "REGISTRY_HTTP_ADDR=0.0.0.0:5003"
  ]

  restart = "unless-stopped"
}

# Create a network for the containers
resource "docker_network" "churn_network" {
  name = "churn-network"
}

# Create MLflow container
resource "docker_container" "mlflow" {
  name  = "mlflow-server"
  image = docker_image.mlflow.image_id

  networks_advanced {
    name = docker_network.churn_network.name
  }

  ports {
    internal = 5000
    external = 5002  # 5000 was occupied by Apply AirPlay on my local
  }

  volumes {
    container_path = "/mlflow/mlruns"
    host_path      = "${path.cwd}/mlflow/mlruns"
  }

  volumes {
    container_path = "/mlflow/mlflow.db"
    host_path      = "${path.cwd}/mlflow/mlflow.db"
  }

  working_dir = "/mlflow"

  command = [
    "mlflow", "server",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "file:///mlflow/mlruns",
    "--host", "0.0.0.0",
    "--port", "5000"
  ]

  restart = "unless-stopped"
}

# Create Airflow server container
resource "docker_container" "airflow_server" {
  name  = "airflow-server"
  image = docker_image.airflow.image_id

  networks_advanced {
    name = docker_network.churn_network.name
  }

  ports {
    internal = 8080
    external = 8081
  }

  volumes {
    container_path = "/opt/airflow/dags"
    host_path      = "${path.cwd}/airflow/dags"
  }

  volumes {
    container_path = "/opt/airflow/logs"
    host_path      = "${path.cwd}/airflow/logs"
  }

  volumes {
    container_path = "/opt/airflow/airflow.db"
    host_path      = "${path.cwd}/airflow/airflow.db"
  }

  volumes {
    container_path = "/opt/airflow/data"
    host_path      = "${path.cwd}/data"
  }

  volumes {
    container_path = "/opt/airflow/models"
    host_path      = "${path.cwd}/models"
  }

  env = [
    "AIRFLOW__CORE__EXECUTOR=LocalExecutor",
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db",
    "AIRFLOW__CORE__LOAD_EXAMPLES=false",
    "AIRFLOW__DAGS__ARE_PAUSED_AT_CREATION=false",
    "KAGGLE_USERNAME=${var.kaggle_username}",
    "KAGGLE_KEY=${var.kaggle_key}"
  ]

  command = ["airflow", "standalone"]

  restart = "unless-stopped"
}

resource "docker_image" "registry" {
  name = "registry:3.0.0"
}

resource "docker_image" "mlflow" {
  name = "ghcr.io/mlflow/mlflow:v3.2.0"
}

resource "docker_image" "airflow" {
  name = "churn-airflow:latest"
  build {
    context = "."
    dockerfile = "Dockerfile.airflow"

    # Force rebuild when source changes
    no_cache = true
  }

  # Force rebuild when dependencies change
  triggers = {
    dockerfile = filemd5("Dockerfile.airflow")
    requirements = filemd5("requirements.txt")
    source = filemd5("pyproject.toml")
  }
}
