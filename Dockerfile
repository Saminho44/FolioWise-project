FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY FolioWise FolioWise
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile

CMD uvicorn FolioWise.api.fast:app --host 0.0.0.0 --port $PORT
