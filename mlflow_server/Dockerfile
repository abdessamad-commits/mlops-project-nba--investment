FROM ubuntu:20.04

# Install python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory
WORKDIR /mlflow_server

# Install MLFlow
RUN pip3 install mlflow==1.7.2


# Run the MLFlow server
CMD mlflow server --host 0.0.0.0 --backend-store-uri /mlflow_server/

