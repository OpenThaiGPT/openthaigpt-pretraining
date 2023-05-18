# Pretraining

## Environment Setup
### Docker
1. Install Docker 
```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```
2. Setup Nvidia Repository
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
3. Install Nvidia-Container Toolkit
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Vscode
1. Install [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. `Ctrl + Shift + P` (Windows) or `CMD + Shift + P` (Mac) to open editor commands
3. Select `Dev Containers: Reopen in Container`

#### Docker Run
- `docker build -t openthai-gpt && docker run --gpus all openthai-gpt <command>`

### Slurm

## Installation

### Install Core Depedencies

```bash
pip install -r requirements.txt
pre-commit install
pip install -e ./src/core
```

### Install modules you want to work on

```bash
pip install -e ./src/<module_name>
```

```bash
pip install -e ./src/data
pip install -e ./src/model
pip install -e ./src/evaluation
```

## Testing

To Run Tests on Development Environment

```bash
pytest tests/data
pytest tests/model
pytest tests/evaluation
pytest tests/core
```
