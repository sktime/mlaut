FROM ubuntu


RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y python3-tk  
RUN mkdir -p /mlaut/mlaut
RUN mkdir -p mlaut/data
ADD ./requirements.txt /mlaut
RUN pip install -r /mlaut/requirements.txt
WORKDIR /mlaut