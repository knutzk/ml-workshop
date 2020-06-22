FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

WORKDIR /data

COPY requirements.txt /data

RUN pip install -r requirements.txt

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
