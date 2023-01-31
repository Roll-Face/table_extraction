FROM ubuntu:18.04

WORKDIR /table_extraction
COPY . /table_extraction

RUN apt-get update -y && apt-get install -y python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html 
RUN pip3 install -r installation/requirements.txt

CMD /bin/bash