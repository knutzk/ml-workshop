FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

WORKDIR /data

EXPOSE 8888

COPY requirements.txt /data

RUN pip install -r requirements.txt

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "-y"]
