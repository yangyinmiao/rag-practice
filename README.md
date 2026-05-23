# RAG 实战练习

基于 LlamaIndex + Chroma + BGE + DeepSeek 的本地 RAG 系统，用于实战学习 RAG 核心流程。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 框架 | LlamaIndex | RAG 全流程编排 |
| 向量数据库 | Chroma | 本地持久化 |
| Embedding 模型 | bge-m3 | 中英文，本地 MPS 加速 |
| Rerank 模型 | bge-reranker-v2-m3 | 与 bge-m3 配套，本地运行 |
| LLM | DeepSeek API | 中文效果好，国内低延迟 |
| 评估框架 | RAGAS | Faithfulness + Answer Relevance |
| 运行环境 | macOS M2 Pro 32GB | MPS 加速推理 |

## 项目结构

```
rag-practice/
├── docs/                  # 原始文档（PDF、txt）
├── src/
│   ├── __init__.py
│   ├── ingest.py          # 文档加载、chunking、embedding 入库
│   ├── retriever.py       # 向量召回 + rerank
│   ├── generator.py       # 组装 prompt + 调 DeepSeek API
│   └── pipeline.py        # 串联检索和生成，对外暴露 query 接口
├── evaluate/
│   ├── qa_pairs.json      # 手写问答对（20 条）
│   └── run_ragas.py       # RAGAS 评估脚本
├── storage/
│   └── chroma/            # Chroma 向量库持久化（自动生成，不入 git）
├── main.py                # 入口
├── .env                   # API Key（不入 git）
├── requirements.txt
├── TODO.md
└── README.md
```

## 快速开始

**1. 克隆项目，激活虚拟环境**

```bash
cd rag-practice
source venv/bin/activate
```

**2. 配置 API Key**

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

**3. 文档入库**

```bash
python src/ingest.py
```

**4. 启动问答**

```bash
python main.py
```

**5. 跑评估**

```bash
python evaluate/run_ragas.py
```

## RAG 流程

```
用户 query
   ↓
bge-m3 embedding
   ↓
Chroma 向量召回 top-50
   ↓
bge-reranker-v2-m3 重排，取 top-5
   ↓
组装 prompt（原始问题 + top-5 文档片段）
   ↓
DeepSeek API 生成回答
   ↓
返回用户
```

## 评估指标

- **Faithfulness**：回答内容是否来自检索到的文档，有没有幻觉
- **Answer Relevance**：回答是否真正回答了用户的问题
- **Context Relevance**：检索出的文档片段是否和问题相关

## 环境要求

- macOS Apple Silicon（MPS 加速）
- Python 3.11
- 内存 16GB+（推荐 32GB）
