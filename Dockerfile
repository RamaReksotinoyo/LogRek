FROM python:3.10-slim as base

RUN pip3 install poetry==1.7.1
WORKDIR /app
COPY pyproject.toml poetry.lock /app

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "main.py"]