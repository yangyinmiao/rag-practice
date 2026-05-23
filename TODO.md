# RAG 实战任务清单

## 阶段一：环境搭建
- [x] 创建项目目录 + Python venv（3.11）
- [x] 安装依赖：llama-index、chromadb、torch（MPS支持）、transformers、sentence-transformers、FlagEmbedding
- [x] 验证 MPS 可用：torch.backends.mps.is_available()

## 阶段二：文档入库
- [x] 准备测试文档（先用几个 PDF 或 txt，别一上来就堆几百个）
- [ ] 用 LlamaIndex 做文档加载 + chunking
- [ ] 用 bge-m3 做 embedding，写入 Chroma

## 阶段三：检索链路
- [ ] 实现 query embedding → Chroma 召回 top-50
- [ ] 接 bge-reranker-v2-m3 重排，取 top-5
- [ ] 组装 prompt，调 DeepSeek API 返回答案

## 阶段四：评估
- [ ] 手写 20 条问答对
- [ ] 接 RAGAS 跑 Faithfulness + Answer Relevance
- [ ] 根据结果找瓶颈，调 chunk size 或 top-k
