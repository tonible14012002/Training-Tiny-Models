
export PYTHONPATH=./

start:
	python app/main.py

run-cmd:
	python cmd/$(file).py