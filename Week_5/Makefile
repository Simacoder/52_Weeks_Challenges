OS := $(shell uname 2>/dev/null || echo Windows)
ifeq ($(OS), Windows)
	PYTHON = python
	PIP = pip
else
	PYTHON = python3
	PIP = pip3
endif

setup:
	$(PYTHON) -m venv mlop_env
	mlop_env/Scripts/python -m pip install --upgrade pip
	mlop_env/Scripts/pip install -r requirements.txt

run:
	mlop_env/Scripts/python main.py

mlflow:
	mlop_env/Scripts/python -m mlflow ui

test:
	mlop_env/Scripts/python -m pytest

clean:
	rm -rf steps/__pycache__ __pycache__ .pytest_cache tests/__pycache__

remove:
	rm -rf mlop_env
