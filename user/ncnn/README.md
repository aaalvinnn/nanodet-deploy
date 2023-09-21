# NanoDet NCNN

This project provides NanoDet image inference using
[Tencent's NCNN framework](https://github.com/Tencent/ncnn).

# Environment

```
ncnn
cmake
nanodet-deploy
```

## Windows

### Step1.

Download and Install Visual Studio from https://visualstudio.microsoft.com/vs/community/

### Step2.

Download and install OpenCV from https://github.com/opencv/opencv/releases

### Step3 (Optional).

Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step4.

Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```

Build NCNN following this tutorial: [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)

### Step5.

Add `ncnn_DIR` = `YOUR_NCNN_PATH/build/install/lib/cmake/ncnn` to system environment variables.

Build project: Open x64 Native Tools Command Prompt for VS 2019 or 2017

``` cmd
mkdir -p build
cd build
cmake ..
msbuild nanodet_demo.vcxproj /p:configuration=release /p:platform=x64
```

## Linux

### Step1.

Build and install OpenCV from https://github.com/opencv/opencv

### Step2(Optional).

Download Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step3.

Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```

Build NCNN following this tutorial: [Build for Linux / NVIDIA Jetson / Raspberry Pi](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)

### Step4.

Set environment variables. Run:

(Remember `make install` to install `ncnn`, otherwise there\`s no path called `/install/lib/cmake/ncnn`)

``` shell script
export ncnn_DIR=YOUR_NCNN_PATH/build/install/lib/cmake/ncnn
```

Build project

``` shell script
mkdir build
cd build
cmake ..
make
```

## Inference images

```shell script
./nanodet 1 ${IMAGE_FOLDER}/*.jpg
```

****

# Custom model

## Export to ONNX

```shell script
python tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
```

## Convert to ncnn

Run **onnx2ncnn** in ncnn tools to generate ncnn .param and .bin file.

After that, using **ncnnoptimize** to optimize ncnn model.

If you have quentions about converting ncnn model, refer to ncnn wiki. https://github.com/Tencent/ncnn/wiki

You can also convert the model with an online tool https://convertmodel.com/ .

## Modify hyperparameters

If you want to use custom model, please make sure the hyperparameters
in `nanodet.h` are the same with your training config file.

```cpp
int input_size[2] = {320, 320}; // input height and width
int num_class = 80; // number of classes. 80 for COCO
int reg_max = 7; // `reg_max` set in the training config. Default: 7.
std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
```

## Build

1. ```
   cd user/ncnn
   ```

2. ```
   mkdir build
   ```

3. ```
   cd build
   cmake ..
   make
   ```

4. ```
   ./nanodet [YOUR_IMAGE_PATH]
   ```

- generated files include:
  - `output`: where saving the inferred results.
