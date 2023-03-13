COUNT_EYE := 30
COUNT_GAUSS := 30
COUNT_MOTION := 30

DB_NAME := 'results'
TABLE_NAME := 'all_models'
MODELS := usrnet kerunc dwdn wiener_blind_noise wiener_nonblind_noise
CONFIG := 'configs/config.yml'


.PHONY: install
install:
	pip install -r requirements_old_torch.txt


.PHONY: prepare_raw_data
prepare_raw_data:
	python src/data/prepare/eye.py $(COUNT_EYE)
	python src/data/prepare/levin.py
	python src/data/prepare/sun.py
	python src/data/generate/gauss_blur.py $(COUNT_GAUSS)
	python src/data/generate/motion_blur.py $(COUNT_MOTION)
	python src/data/generate/benchmark_list.py
	echo 'All datasets are prepared!'

.PHONY: test
test:
	python src/main.py --config $(CONFIG) --models $(MODELS) --db_name $(DB_NAME) --table_name $(TABLE_NAME)

.PHONY: test_help
test_help:
	python src/main.py --help

.PHONY: build
build:
	docker build . -f Dockerfile -t nbdb-torch1.7.1 --force-rm

.PHONY: run
run:
	docker run --runtime=nvidia -it --name nbdb-c --mount type=bind,source=./datasets,target=/nbdb/datasets,bind-propagation=rslave --mount type=bind,source=./results,target=/nbdb/results,bind-propagation=rslave --entrypoint=/bin/bash nbdb-torch1.7.1

.PHONY: exec
exec:
	docker exec -it nbdb-c bash