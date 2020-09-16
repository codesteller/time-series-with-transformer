# Time Series Forecasting with Transformers
With Pytorch NGC Container - Pull Tag - docker pull nvcr.io/nvidia/pytorch:20.07-py3

# Multiple GPU Training
python transformer-multigpu.py --gpu_devices 0 1 --batch_size 200

# Single GPU Training
python transformer-multigpu.py --gpu_devices 0 --batch_size 200

