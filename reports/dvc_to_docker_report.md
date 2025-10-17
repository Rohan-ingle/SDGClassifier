# SDG Classifier MLOps Report

## Overview
This report summarizes the end-to-end workflow implemented for the SDG Classifier project, covering data versioning with DVC, continuous integration through GitHub Actions, and containerization with Docker for deployment.

## Data Version Control (DVC)
- The pipeline is defined in `dvc.yaml` with five stages: `data_preprocessing`, `train_model`, `evaluate_model`, `model_validation`, and `export_model`.
- Each stage runs a Python entry point under `src/` and captures its dependencies, parameters from `params.yaml`, and materialized artifacts under `data/`, `models/`, and `metrics/`.
- Remote storage is configured in `.dvc/config` to use the `mlopsl-sdgclassifier` S3 bucket (`s3://mlopsl-sdgclassifier/dvc`).
- To reproduce and publish artifacts:
  1. Install dependencies with `pip install -r requirements.txt`.
  2. Configure AWS credentials locally (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) and set up the S3 remote with `dvc remote default s3remote` if needed.
  3. Run `dvc pull` to retrieve cached artifacts or `dvc repro` to regenerate processed data, models, metrics, and export bundles.
  4. Inspect outcomes via `dvc metrics show` and review plots stored in `metrics/`.
  5. Publish new artifacts with `dvc push`, which uploads to the S3 remote.
- The `plots` and `metrics` sections of `dvc.yaml` allow DVC to track evaluation artifacts over time and compare runs with `dvc metrics diff`.

## Continuous Integration and Delivery (CI/CD)
- Automated checks are configured in `.github/workflows/ci-cd.yaml`.
- The `quality` job runs on every push or pull request to `main` or `develop` and performs:
  - Dependency installation with cached wheels.
  - Linting via `flake8` over `src` and `tests`.
  - Unit testing using `pytest` with warnings treated as non-fatal.
  - A structural DVC inspection using `dvc dag --full`.
  - Artifact upload for cached pytest data and metric snapshots.
- When GitHub secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` are supplied, the workflow authenticates with AWS and runs `dvc push` to publish artifacts to the S3 remote automatically.
- The `docker` job depends on successful quality checks and builds the project image using `docker/build-push-action` to ensure the Dockerfile remains valid. Image publishing can be enabled later by adding registry credentials and setting `push: true`.

## Docker Containerization
- The runtime environment is specified in `Dockerfile` using `python:3.10-slim` as the base layer.
- Core steps performed:
  - System packages for scientific Python builds are installed (`build-essential`, `gcc`).
  - Project dependencies are installed from `requirements.txt` with pip caching disabled for smaller layers.
  - Application source, trained artifacts, and configuration files are copied into `/app`.
  - Streamlit environment variables expose the UI on port `8501`; the container exposes the same port.
  - Container startup runs `streamlit run app.py`, launching the SDG classifier interface.
- To build and run the container locally:
  1. `docker build -t sdg-classifier:latest .`
  2. `docker run --rm -p 8501:8501 sdg-classifier:latest`
  3. Access the UI at `http://localhost:8501` and ensure `models/inference_pipeline.pkl` is present in the image or mounted via volume.

## End-to-End Hosting Steps
1. **Data preparation**: Use DVC to pull cached data (`dvc pull`) or regenerate artifacts (`dvc repro`) before model updates.
2. **Metric review**: Inspect outputs in `metrics/` and compare with previous runs via `dvc metrics diff`.
3. **Quality gates**: Push commits to GitHub to run linting, tests, DVC checks, and conditional artifact publishing through the CI pipeline.
4. **Image build**: Use the validated Dockerfile to build images locally or in CI (`docker build`). Push to a registry when release-ready (`docker push`).
5. **Cloud synchronization**: Ensure the latest DVC cache is uploaded with `dvc push`, allowing AWS hosts to sync artifacts from S3.
6. **Deployment**: Launch the container on AWS or other infrastructure with persistent volume access to models if updates are expected.

## AWS Deployment Blueprint
- The CloudFormation template `infra/ec2-docker-template.yaml` provisions an Ubuntu-based EC2 instance, security group, IAM role, and instance profile with S3 and ECR read access.
- User data on boot performs: system updates, Docker installation, repository checkout, S3 sync of the DVC cache, local Docker image build, and container launch exposing port 80 (mapped to Streamlit’s 8501).
- Parameters allow customization of instance type, VPC/subnet placement, Git repository, branch, S3 bucket, and Docker tag.
- To deploy:
  1. Upload the template in CloudFormation and provide required parameters (key pair, VPC, subnet, repo URL, etc.).
  2. Ensure the target bucket (`mlopsl-sdgclassifier`) contains the latest DVC cache via `dvc push` before stack creation.
  3. After stack completion, access the application using the output public IP/DNS on port 80.

## Security and Secrets Management
- For CI, store AWS credentials as GitHub Secrets to enable automated `dvc push` without embedding keys in code.
- On EC2, the instance role granted by the template provides least-privilege access to S3 and ECR; avoid hard-coding credentials on the host.
- Rotate credentials regularly and monitor S3 bucket policies to restrict access to required principals only.

## Maintenance Checklist
- Update `params.yaml` to tune preprocessing or model hyper-parameters; rerun `dvc repro` to capture new artifacts.
- Keep `requirements.txt` aligned with runtime needs and review dependency updates regularly.
- Extend `.github/workflows/ci-cd.yaml` with additional jobs (security scans, integration tests) as the project evolves.
- Version Docker images and maintain release notes so downstream environments can track model and code builds.
