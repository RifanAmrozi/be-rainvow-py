run:
	uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload --reload-dir app

install:
	pip install -r requirements.txt

format:
	black .

test:
	pytest -v

db-migrate:
	alembic upgrade head

ip:
	ipconfig getifaddr en0

mtx:
	./mediamtx mediamtx.yml

env:
	source .venv/bin/activate
