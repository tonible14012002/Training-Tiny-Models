

# Run file passed by user
run-cmd:
	export PYTHONPATH="./"
	python cmd/$(file)
