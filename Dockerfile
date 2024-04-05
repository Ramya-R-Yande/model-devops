FROM python:3.12.0
COPY . /app
WORKDIR /app
RUN pip install -v -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app