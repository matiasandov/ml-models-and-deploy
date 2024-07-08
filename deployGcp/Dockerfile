FROM python:3.8-slim

#se crea un dir para ejecutar app
WORKDIR /app

#copiar file del directorio actual en app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

#COMANDOS QUE SE CORRERAN
CMD ["python", "app.py"]