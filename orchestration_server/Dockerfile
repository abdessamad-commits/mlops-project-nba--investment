FROM apache/airflow:latest-python3.9

# Copy requirements file
COPY requirements.txt /requirements.txt

# switch to root user
USER root

# Install build tools
RUN apt-get update && apt-get install -y build-essential

# switch back to non-root user
USER airflow

RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install --user --upgrade pip

# Install MLflow 
RUN pip3 install --no-cache-dir --user mlflow

# Install the rest of the dependencies
RUN pip3 install --no-cache-dir --user -r /requirements.txt
