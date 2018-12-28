FROM ubuntu
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip install mlaut

RUN mkdir -p /mlaut/mlaut
RUN mkdir -p mlaut/data

WORKDIR /mlaut