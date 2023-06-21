FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3.11-dev build-essential
RUN curl -sSL https://install.python-poetry.org | python3 -
WORKDIR /app
RUN poetry config virtualenvs.create false
COPY pyproject.toml pyproject.toml
RUN poetry install --no-interaction --no-ansi
COPY ./services ./services
COPY ./models ./models
ENTRYPOINT ["python3.11"]
CMD ["services/server_221785.py"]