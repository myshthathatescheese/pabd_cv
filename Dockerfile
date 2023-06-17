FROM tensorflow/tensorflow:2.12.0
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./services ./services
COPY ./models ./models
ENTRYPOINT ["python"]
CMD ["services/server_221234.py"]
