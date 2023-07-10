FROM python:3.10-slim

COPY ./inputData/test_run/test_1 /input/test_1/
COPY ./src /src/
COPY ./app /app
COPY requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt
RUN apt-get update
RUN apt-get install libgomp1

CMD [""]