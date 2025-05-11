FROM python:3.12-slim

WORKDIR /

RUN apt-get update && apt-get install -y git

COPY . /

RUN pip install --no-cache-dir -r requirements.txt

# setup an app user so the container does not run as the root user
RUN useradd app
USER app

EXPOSE 8080

CMD ["python", "flask_api.py"]