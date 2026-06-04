FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .



COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python download_model.py

ENV FLASK_APP=api_services.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

EXPOSE 8080

CMD ["python", "api_services.py"]
