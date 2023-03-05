# Base image
FROM python:3.9-slim-buster

WORKDIR /app

#Install Git
RUN apt-get update && apt-get install -y git

#upgrade PIP
RUN pip install --upgrade pip

#Clone git repository
RUN git clone https://github.com/quantumdoubleblind/qsw_2023.git

#Install dependencies
ADD  requirements.txt app/

RUN pip install -r app/requirements.txt

CMD ["python", "qsw_2023/custom_topologies_qaoa/src/experiments.py"]