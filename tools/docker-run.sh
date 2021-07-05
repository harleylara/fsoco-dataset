#!/usr/bin/env bash
# Bash script to document the FSOCO tools docker usage 
echo "This is an example script, uncomment and adjust the run command that suits your use-case." 
echo "USAGE: bash docker-run.sh [FSOCO CLI Arguments]"
echo "Arguments passed: $@" 

# For adding further volumes, see: https://docs.docker.com/storage/bind-mounts/
# With GUI, needs XServer
docker run --gpus '"device=0"' \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/usr/app/src/data/ \
    -e DISPLAY=$DISPLAY \
    -u fsoco \
    fsoco/fsoco:latest "$@"
    
# Without GUI, tools that have visual output won't work
#docker run -it fsoco/fsoco:latest --gpus '"device=0"' \
#    -v $PWD:/usr/app/src/data/ \
#    -u fsoco \
#    fsoco/fsoco:latest "$@"
