<<<<<<< Updated upstream
docker run --gpus '"device=1"' --rm -it -p 9888:8888 -p 9006:6006 -v $(pwd):/workspace/time_series_train -w /workspace/time_series_train  nvcr.io/nvidia/tensorflow:20.08-tf2-py3
=======
docker run --gpus '"device=3"' --rm -it -v $(pwd):/workspace/time_series_train -w /workspace/time_series_train  nvcr.io/nvidia/tensorflow:20.03-tf2-py3
>>>>>>> Stashed changes
