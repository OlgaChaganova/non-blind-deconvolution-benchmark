COUNT_EYE := 30
COUNT_GAUSS := 30
COUNT_MOTION := 30

DB_NAME := 'results'
TABLE_NAME := 'all_models'


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
	python src/main.py --models $(MODELS) --db_name $(DB_NAME) --table_name $(TABLE_NAME)