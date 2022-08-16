# syntax=docker/dockerfile:1

FROM python:3.10.5-slim
WORKDIR /colorskim_app
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt
EXPOSE 8501
COPY /colorskim_app .
CMD [ "streamlit", "run", "streamlit-serv.py"]