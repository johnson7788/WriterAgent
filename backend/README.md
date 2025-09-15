# 综述生成的后端的代码

# 安装
pip install -r requirements.txt

# 运行
python main.py

# 目录
main_api #主程序
main_outline #大纲生成
main_content #内容生成
search_api   #搜索API，里面有1个接口


# curl进行测试
curl -X POST http://127.0.0.1:7800/api/review_outline \
  -H "Content-Type: application/json" \
  -d '{"content":"多模态大模型安全性","language":"zh","stream":false}'
