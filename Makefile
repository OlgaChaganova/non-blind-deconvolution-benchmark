COUNT_EYE := 30
COUNT_GAUSS := 30
COUNT_MOTION := 30

DB_NAME := 'results/metrics'
TABLE_NAME := 'all_models'
MODELS := usrnet kerunc dwdn wiener_blind_noise wiener_nonblind_noise
CONFIG := 'configs/config.yml'
MODE := 'main'


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
	python src/main.py --config $(CONFIG) --mode $(MODE) --models $(MODELS) --db_name $(DB_NAME) --table_name $(TABLE_NAME)

.PHONY: test_help
test_help:
	python src/main.py --help

.PHONY: build
build:
	docker stop nbdb-c && docker rm nbdb-c
	docker build . -f Dockerfile -t nbdb --force-rm

.PHONY: run
run:
	docker run -p 5050:5050 --runtime=nvidia -it --name nbdb-c --mount type=bind,source=./datasets,target=/nbdb/datasets,bind-propagation=rslave --mount type=bind,source=./results,target=/nbdb/results,bind-propagation=rslave --mount type=bind,source=./notebooks,target=/nbdb/notebooks --entrypoint=/bin/bash nbdb

.PHONY: exec
exec:
	docker exec -it nbdb-c bash