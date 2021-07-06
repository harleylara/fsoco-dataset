docker run -it --gpus all -v $PWD/:/usr/app/src/data/ --ipc=host fsoco/fsoco:latest fsoco similarity-scorer --gpu 'img_glob/*'
