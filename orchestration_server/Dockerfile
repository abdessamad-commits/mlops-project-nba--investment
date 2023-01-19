FROM apache/airflow:2.0.1

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip3 install --user --upgrade pip
RUN pip3 install --no-cache-dir --user -r /requirements.txt

# Install MLflow 
RUN pip3 install --no-cache-dir --user mlflow

