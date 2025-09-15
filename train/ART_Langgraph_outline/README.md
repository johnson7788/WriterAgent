# Langgraph ART 训练根据topic生成大纲模型

## 训练流程
1. 环境部署： [prepare.md](../prepare.md)
2. 测试下未训练过的模型： [original_model.py](original_model.py)
3. 生成训练样本： [generate_topic.py](generate_topic.py)
4. 开始训练,修改.env文件： [train.py](train.py)
    - 训练时和测试时使用的prompt.py
5. 测试模型训练效果: [model_test.py](model_test.py)

## 文件
```
├── README.md
├── env_template   ##模版文件，使用哪个模型进行搜索和作为reward 模型
├── generate_topic.py  ## 生成训练数据，使用的是Openai的Agent框架和Json的AgentOutputSchemaBase输出
├── model_test.py ## 训练后的模型进行测试
├── prompt.py  #训练时的prompt,生成大纲和评估大纲的奖励模型
├── requirements.txt
├── topic.json    #训练时需要的主题数据
├── original_model.py    #测试未经过训练的模型
└── train.py       # 训练代码
```

## 搜索输出
```
WebSearchClient = ZhipuAiClient(api_key="your-api-key")
response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query="搜索2025年4月的财经新闻",
        count=15,  # 返回结果的条数，范围1-50，默认10
        search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
        content_size="high"  # 控制网页摘要的字数，默认medium
    )
 [
    {
      "title": "【专题报告】股指主题会议纪要",
      "link": "",
      "content": "正文共4948字，阅读时间约10分钟 观点概述： 4月以来，股市走出慢牛”行情，大盘托底反内卷”驱动小盘轮动，带动上证屡创新高。本轮牛市是否可持续，还有多少上涨空间，节奏上如何把握，各品种间又有哪些投资机会，是我们本次股指会议研讨的重点。 2025年4月大跌后，国家队”救市为大盘托底，但前期市场仍聚集在EPS存在改善预期的行业和中证1000等小盘股，指数并未出现明显抬升，7月，美国经济数据走弱，海外降息空间打开；中央财经委员会强调反内卷”， 引发供给侧改革2.0遐想；股市财富效应显现，流动性向股市转移，水牛”出现，指数突破上涨。但国内数据显示经济动能开始走弱，与股市持续上涨形成背离，在居民存款搬家和风险偏好抬升”外，是否还有其他关键因素引导了此次上涨，我们是否处在某个周期性的拐点？ 此次会议后，我们认为空间上，长期还有上涨空间，从水牛”转向盈利牛”是主要方向；节奏上短期未见顶，但上行趋势中指数有波动可能；品种推荐从中证1000转向沪深300和中证500，以垄断性龙头和周期性板块EPS修复为主线。 具体品种： 沪深300。具备全球定价权或高集中度龙头，估值仍处低位，戴维斯双击空间最大；国家队持续托底，融资盘/减持均处低位，供给收缩分红提升带来紧平衡”；海外流动性被动扩张（美债利率高位回落、美元下行），外资回流首选大盘蓝筹。 中证500。周期板块EPS扩张型特征明确，地产链利润下滑已计价，成本端地价见顶回落带来毛利修复；财政扩张若转向有效投资，周期品需求弹性最大，有色、化工等龙头集中在中证500；贴水尚未极致，量化与雪球资金持续增配，收益估值修复双击。 风险：政府加税和地产超预期走弱是需要关注的风险，海外环境也需要持续跟踪，尚不做预测。 一国内外宏观情况分享 1、当前海外宏观环境 货币政策环境，汇率市场波动，基本面和地缘（关税）是4-7月中美股市共振的主要因素。 货币政策方面，中美同步交易降息预期。美国近期经济硬数据有波动，但长期看美国政策端对货币环境引导加强，美联储理事任命受政治影响，鹰派声音减弱，大概率降息；且美国流动性工具储备情况及财政税收季需求，将倒逼美联储采取行动。 汇率市场方面，美元指数年初至今整体走弱，人民币被动企稳。美元指数虽 7 月有小反弹，但长期向下趋势不变。美欧、美日利差缩小，资金回流美国动力不足，且美国债务问题将持续施压美元指数，美元美债背离将可能持续。 基本面方面，中美股市主线均集中于盈利改善的互联网金融，",
      "icon": "",
      "media": "",
      "refer": "ref_1",
      "publish_date": "2025-08-25"
    }
 ]
```

## 训练
python train.py

## 测试
python model_test.py

## wandb日志
http://192.168.100.8:3005/johnson/web-search-agent-training

## 训练的模型结果
```
PROJECT_NAME = web-search-outline-training
ART_NAME=ppt-outline05
/workspace/verl/RLDecisionAgent/ART/.art/web-search-outline-training/models/ppt-outline05/
├── checkpoints
│   ├── 0000
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0001
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0002
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0003
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   └── 0004
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── added_tokens.json
│       ├── chat_template.jinja
│       ├── merges.txt
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── vocab.json
├── history.jsonl
├── logs
│   └── vllm.log
├── model.json
├── tensors
│   ├── advantages.pt
│   ├── assistant_mask.pt
│   ├── group_ids.pt
│   ├── input_pos.pt
│   ├── logprobs.pt
│   ├── parent_ids.pt
│   ├── tokens.pt
│   └── weights.pt
└── trajectories
    └── train
        ├── 0000.jsonl
        ├── 0001.jsonl
        ├── 0002.jsonl
        └── 0003.jsonl
```