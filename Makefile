
export PYTHONPATH=./

start:
	python app/main.py --reload --port 8000

start-dev:
	python app/main.py --reload --port 8003


run-cmd:
	python cmd/$(file).py

deploy:
	python app/main.py --host 0.0.0.0 --port 8000 --workers 4