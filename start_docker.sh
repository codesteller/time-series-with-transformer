docker run --gpus all --rm -it --ipc=host -v $(pwd):/workspace/time_series_train -w /workspace/time_series_train  nvcr.io/nvidia/pytorch:20.03-py3