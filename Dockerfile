# syntax=docker/dockerfile:1

FROM python:3.10.5-slim-buster
WORKDIR /colorskim_app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /colorskim_app .
CMD [ "streamlit", "run", "jpao_library.py"]