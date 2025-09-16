

# Run file passed by user
run-cmd:
	export PYTHONPATH="./"
	python cmd/$(file)

start:
	export PYTHONPATH="./"
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000