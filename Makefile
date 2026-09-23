.PHONY: install run docker-build docker-run

install:
	pip install -r requirements

run:
	uvicorn main:app --reload

docker-build:
	docker build -t books-api .

docker-run:
	docker run -p 8000:8000 books-api
