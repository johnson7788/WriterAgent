# 环境部署

## 智星云上的Huggingface镜像(不好用)
export HF_ENDPOINT=http://192.168.50.202:18090

##  尝试使用Areal的镜像
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name areal ghcr.io/inclusionai/areal-runtime:v0.3.0.post2 sleep infinity
docker start areal
docker exec -it areal bash
# 设置pip镜像源
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
cd ART
pip install .
pip install ".[backend]"
# 设置代理，安装git上的项目
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPs_PROXY=http://127.0.0.1:7890
pip install 'torchtune @ git+https://github.com/pytorch/torchtune.git'
pip install 'unsloth-zoo @ git+https://github.com/bradhilton/unsloth-zoo'
