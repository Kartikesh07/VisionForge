# Deployment guide

VisionForge supports CPU inference without paid APIs or external inference services. The Docker configuration runs Streamlit on port 7860 as a non-root user. A trained checkpoint is required; datasets are optional for the dashboard. The Linux container build has not yet been validated, so verify it before relying on a hosted installation.

## Hugging Face Spaces

Use **Docker** for Streamlit. The root README sets `sdk: docker` and `app_port: 7860`; the image serves Streamlit on `0.0.0.0:7860` as UID 1000. See Hugging Face's [deprecated Streamlit SDK](https://huggingface.co/docs/hub/spaces-sdks-streamlit) and [Docker configuration](https://huggingface.co/docs/hub/spaces-sdks-docker).

Hugging Face's current documentation says new Docker Spaces require a paid account plan. CPU Basic has no hourly compute charge, but account eligibility is separate. Check the [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) for current requirements. Local execution needs no paid service.

## Reliable checkpoint delivery

Model binaries and datasets are excluded from Git and the image. `deployment/model_manifest.json` pins the existing ResNet18 by size and SHA256. Startup accepts either:

1. A local file mounted at `/app/artifacts/models/resnet18_best.pth`.
2. `VISIONFORGE_CHECKPOINT_URL`, an HTTPS URL serving the release checkpoint bytes. A model repository URL has the form `https://huggingface.co/<owner>/<model>/resolve/<commit>/resnet18_best.pth`. Pin an immutable commit. A public release checkpoint URL is not currently available.

The release checkpoint is 44,798,219 bytes with SHA256 `ca32161ed4977e5390bc939d9fb3058da3cdec7b11a3c38cfc8c4ebf7ce0b380`. Startup verifies existing files or downloads into a temporary file, validates size/hash and publishes atomically. Incorrect existing files are never overwritten; failed downloads are removed and startup fails clearly. Obtain the checkpoint from the maintainer before following the release deployment instructions.

A public checkpoint URL needs no API key. Private authenticated model hosting is not implemented; do not place credentials in URLs, Git or logs. For a separately trained model, update the manifest to its verified size/hash and supply its own evaluation report; release metrics do not apply to replacement weights. Review dataset/provider and pretrained-weight redistribution terms before distributing weights.

## Local Docker

Install Docker and place the release checkpoint under `artifacts/models/`. Run from the repository root. Python inside the image uses `/opt/venv/bin/python`.

```powershell
docker build -t visionforge:cpu .
$checkpoint = (Resolve-Path .\artifacts\models\resnet18_best.pth).Path
docker run --rm -p 7860:7860 --mount "type=bind,source=$checkpoint,target=/app/artifacts/models/resnet18_best.pth,readonly" visionforge:cpu
```

Linux/macOS:

```bash
docker build -t visionforge:cpu .
docker run --rm -p 7860:7860 --mount "type=bind,source=$(pwd)/artifacts/models/resnet18_best.pth,target=/app/artifacts/models/resnet18_best.pth,readonly" visionforge:cpu
```

Open `http://localhost:7860`. There is no dataset in the image, so local validation samples are disabled. The health endpoint is `http://localhost:7860/_stcore/health`.

The image installs official CPU PyTorch/TorchVision wheels, pinned runtime dependencies and OpenCV system libraries. It forces CPU inference with two OpenMP threads. `.dockerignore` allows only source, Streamlit configuration and deployment files. `deployment/evaluation.json` contains the release evaluation results with repository-relative checkpoint paths, making metrics available without local report directories.

## Hugging Face configuration

1. Verify the Docker image locally with the matching checkpoint.
2. Make the checkpoint accessible through an HTTPS download URL, respecting redistribution terms.
3. Create a Docker Space and upload the repository source and configuration.
4. Set the Space variable `VISIONFORGE_CHECKPOINT_URL` to the immutable checkpoint URL. The image already sets `VISIONFORGE_DEVICE=cpu`.
5. Confirm startup and health, then test uploads, six-class probabilities, Grad-CAM and model performance navigation.

The source repository excludes environments, datasets, caches, temporary test directories and model binaries. Only the compact evaluation JSON and explanatory image are included as benchmark documentation.
