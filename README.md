# nanodet-deploy
This project is for deployment of nanodet-m onnx model, including onnxruntime, ncnn...

## How to Use

### Requirements

- CUDA = 11.3
- Python = 3.8
- Pytorch = 1.10.1
- torchvision==0.11.2

### Env

1. Create a conda virtual environment and then activate it.

```
 conda create -n nanodet python=3.8 -y
 conda activate nanodet
```

2. Install pytorch

```
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
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

#### onnxruntime

exam.

```
python .\user\onnxruntime\detect.py image --config .\config\nanodet_custom_xml_dataset.yml --model .\models\nanodet.onnx --path .\user\images\image1.jpg --save_result
```

while `--save_result` is optional.

## Thanks

- https://github.com/RangiLyu/nanodet.git
