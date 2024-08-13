# How to build with intel GPU

## build only cpu version


```bash
mkdir build
cd build
cmake ..
make -j
```

## download model from model zoo

The converted model is on https://huggingface.co/huangyuyang

## run with model

```bash
./fastllm/build/main -p /local/path/models/fast/Qwen-7B-Chat-fp16.flm
```
