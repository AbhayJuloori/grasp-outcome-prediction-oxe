PY ?= .venv/bin/python
export PYTHONPATH := src
export OMP_NUM_THREADS := 1

.PHONY: setup data ingest warehouse train uci explain report dashboard test all

setup:
	uv venv -p 3.11 .venv
	uv pip install -p .venv -r requirements.txt

data:
	scripts/download_data.sh

ingest:
	$(PY) -m grasp.ingest

warehouse:
	$(PY) -m grasp.warehouse

train:
	$(PY) -m grasp.train_kuka

uci:
	$(PY) -m grasp.uci

explain:
	$(PY) -m grasp.explain

report:
	$(PY) -m grasp.report

dashboard:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

all: data ingest warehouse train uci explain report
