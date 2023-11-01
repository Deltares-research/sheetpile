FROM python:3.10-slim

COPY ./src /src/
COPY ./app /app
COPY ./model /model
COPY requirements.txt ./requirements.txt

RUN apt-get -y update
RUN apt-get -y install git
RUN pip install -r ./requirements.txt
RUN apt-get install libgomp1
RUN apt-get -y install libglu1 libgl1 libxrender1 libxcursor1 libxft-dev libxinerama-dev

CMD ["python", "-u", "./src/run_kratos.py"]