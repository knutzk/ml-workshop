FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

WORKDIR /data

EXPOSE 8888

COPY requirements.txt /data

RUN pip install -r requirements.txt

RUN \
  apt-get update -y && \
  apt-get install graphviz -y && \
  apt-get --purge remove -y .\*-doc$ && \
  apt-get clean -y && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/*

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "-y"]
