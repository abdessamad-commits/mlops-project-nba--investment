# 
FROM python:3.9

# 
WORKDIR /model_deployment_server

# 
COPY ./requirements.txt /model_deployment_server/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /model_deployment_server/requirements.txt

# 
COPY ./src /model_deployment_server/src

# 
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "4100"]