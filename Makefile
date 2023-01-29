COUNT_EYE := 30
COUNT_GAUSS := 30
COUNT_MOTION := 30


.PHONY: install
install:
	pip install -r requirements.txt


.PHONY: prepare_raw_data
prepare_raw_data:
	python src/data/prepare/eye.py $(COUNT_EYE)
	python src/data/prepare/levin.py
	python src/data/prepare/sun.py
	python src/data/generate/gauss_blur.py $(COUNT_GAUSS)
	python src/data/generate/motion_blur.py $(COUNT_MOTION)
	echo 'All datasets are prepared!'