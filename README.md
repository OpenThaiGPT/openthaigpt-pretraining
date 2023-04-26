# Pretraining

## Installation

### Install Core Depedencies
```bash
pip install -r requirements.txt
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
pip install -e ./tests/<module_name>
```
```bash
pytest tests/data
pytest tests/model
pytest tests/evaluation
pytest tests/core
```
