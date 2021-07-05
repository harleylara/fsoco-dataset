FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:$PATH"
# Install dependencies
RUN apt-get update && \
  apt-get install -y software-properties-common wget && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y python3.7 python3.7-dev python3.7-venv python3-pip && \
  apt-get install -y ffmpeg libsm6 libxrender1 libfontconfig1 libxtst6 libxi6 git
# Container cwd
WORKDIR /usr/app/src
# Prep FSOCO tools
COPY ./ /usr/app/src/
# Prep Python
ENV VIRTUAL_ENV=/opt/venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install requirements
RUN pip install -e .[sly]
# Add fsoco user
RUN adduser --quiet --disabled-password fsoco
RUN usermod -d /usr/app/src fsoco
RUN chmod -R 777 /usr/app/src
RUN chown -R fsoco /usr/app/src
ENV QT_X11_NO_MITSHM=1

CMD fsoco
