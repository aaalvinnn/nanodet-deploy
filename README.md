# nanodet-deploy
This project is for deployment of nanodet-m onnx model, including onnxruntime, ncnn...

## How to Use

### Requirements

- CUDA = 11.7
- Python = 3.8
- Pytorch = 1.13.1

### Env

1. Create a conda virtual environment and then activate it.

```
 conda create -n nanodet python=3.8 -y
 conda activate nanodet
```

2. Install pytorch

```
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
```

3. Clone this repository

```
git clone https://github.com/aaalvinnn/nanodet-deploy.git
cd nanodet-deploy
```

4. Install requirements

```
pip install -r requirements.txt
```

5. Setup NanoDet

```
python setup.py develop
```

### Infer

