<div align="center">

# 育儿知识库

**面向家长端的家庭教育育儿助手 · 基于 RAG 的智能知识库系统**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-00A1EA)](https://milvus.io/)
[![BGE--M3](https://img.shields.io/badge/Embedding-BGE--M3-5B8FF9)](https://huggingface.co/BAAI/bge-m3)
[![Status](https://img.shields.io/badge/Status-Active-22a06b)](#)

系统接入**育儿建议、专家建议、亲子案例、沟通话术、知识科普**五类内容，提供育儿知识推荐、案例检索、问题问答与知识科普能力。

支持**多轮对话**、**流式输出**与**来源溯源**。

</div>

---

## 目录

- [1. 功能特性](#1-功能特性)
- [2. 技术栈](#2-技术栈)
- [3. 系统架构](#3-系统架构)
- [4. 项目结构](#4-项目结构)
- [5. 快速开始](#5-快速开始)
- [6. 接口说明](#6-接口说明)
- [7. 流水线设计](#7-流水线设计)
- [8. 数据规范](#8-数据规范)
- [9. 向量库 Schema](#9-向量库-schema)
- [10. 关键设计说明](#10-关键设计说明)
- [11. 常见问题](#11-常见问题)
- [12. 后续扩展方向](#12-后续扩展方向)

---

## 1. 功能特性

| 能力 | 说明 |
| :--- | :--- |
| **内容导入** | 支持 PDF / Markdown；自动抽取元数据（内容类型 / 年龄段 / 问题类型 / 场景描述）；导入过程节点级进度追踪 |
| **混合检索** | BGE-M3 生成稠密 + 稀疏双向量，一次编码两种表示；按元数据条件过滤缩小检索范围 |
| **多路召回** | 常规语义检索 + HyDE 假设性文档检索，两路并行召回 |
| **结果融合** | RRF（Reciprocal Rank Fusion）融合多路结果，规避不同检索器分数量纲不可比的问题 |
| **结果精排** | bge-reranker-large 交叉编码器对候选切片精排，输出 Top-K |
| **知识问答** | 四段式回答：详细解释 → 方法建议 → 沟通话术 → 注意事项，并附来源溯源 |
| **交互能力** | 单轮问答、多轮对话（指代消解）、SSE 流式输出、历史记录管理 |
| **健壮性** | 无条件召回兜底、精排失败降级、元数据 LLM 兜底、导入幂等（同文档重导即替换） |

---

## 2. 技术栈

| 层级 | 选型 | 用途 |
| :--- | :--- | :--- |
| Web 框架 | FastAPI + Uvicorn | 提供 REST 接口与 SSE 流 |
| 流程编排 | LangGraph（StateGraph） | 编排两条 RAG 流水线 |
| 向量模型 | `BAAI/bge-m3` | 稠密 + 稀疏双向量生成 |
| 精排模型 | `BAAI/bge-reranker-large` | 交叉编码器精排 |
| 向量数据库 | Milvus | HNSW + SPARSE_INVERTED_INDEX 混合检索 |
| 会话存储 | MongoDB | 多轮对话历史 |
| 对象存储 | MinIO | 存放 PDF 解析出的图片 |
| 文档解析 | MinerU | PDF → Markdown |
| 大模型 | 通义千问（OpenAI 兼容接口） | 条件抽取 / HyDE / 答案生成 |
| 日志 | loguru | 控制台 + 文件双输出 |
| 依赖管理 | uv | 环境与依赖 |

---

## 3. 系统架构

```mermaid
flowchart LR
    subgraph client["前端页面"]
        P1["import.html<br/>文档导入"]
        P2["chat.html<br/>智能问答"]
    end

    subgraph svc["后端服务（两个独立进程）"]
        S1["导入服务 :8000<br/>LangGraph 导入流水线"]
        S2["查询服务 :8001<br/>LangGraph 查询流水线"]
    end

    subgraph store["存储层"]
        M1[("Milvus<br/>向量库")]
        M2[("MongoDB<br/>会话历史")]
        M3[("MinIO<br/>图片对象")]
    end

    subgraph model["模型层"]
        E1["BGE-M3<br/>向量模型"]
        E2["bge-reranker<br/>精排模型"]
        E3["Qwen<br/>大模型"]
    end

    P1 --> S1
    P2 --> S2
    S1 --> E1
    S1 --> M1
    S1 --> M3
    S2 --> E1
    S2 --> E2
    S2 --> E3
    S2 --> M1
    S2 --> M2

    classDef page fill:#e7f0ff,stroke:#2f7aed,color:#1f3d5c
    classDef service fill:#f7f9fb,stroke:#8a99a8,color:#2c3e50
    classDef db fill:#e6f6ef,stroke:#22a06b,color:#1c5c40
    classDef mdl fill:#fff8e8,stroke:#e0932f,color:#7a5a14

    class P1,P2 page
    class S1,S2 service
    class M1,M2,M3 db
    class E1,E2,E3 mdl
```

> **设计要点**：导入与查询拆成两个独立进程，分别占用 `8000` / `8001` 端口。导入是重计算任务、查询是低延迟任务，两者资源特征不同，拆开后可以独立重启与扩容。

---

## 4. 项目结构

```
parenting-kb/
├── .env.example                 # 环境变量模板（复制为 .env 后填写）
├── pyproject.toml               # 依赖声明
├── prompts/                     # 提示词模板
│   ├── metadata_extract.prompt  # 元数据抽取（LLM 兜底）
│   ├── filter_extract.prompt    # 检索条件提取 + 问题改写
│   ├── hyde_prompt.prompt       # 假设性文档生成
│   ├── answer_out.prompt        # 四段式答案生成
│   └── image_summary.prompt     # 图片摘要（PDF 用）
├── doc/                         # 待导入的原始素材（自行放入）
├── logs/                        # 运行日志（自动生成）
├── output/                      # 导入产物：解析结果 / 切片备份 backup.json
└── app/
    ├── conf/                    # 配置层
    │   ├── domain_config.py     # 领域常量（字段名 / 白名单）—— 唯一数据源
    │   ├── lm_config.py         # 大模型配置
    │   ├── embedding_config.py  # 向量模型配置
    │   ├── reranker_config.py   # 精排模型配置
    │   ├── milvus_config.py     # Milvus 配置
    │   ├── minio_config.py      # MinIO 配置
    │   └── mineru_config.py     # MinerU 配置
    ├── core/                    # 框架基础设施
    │   ├── logger.py            # 日志 + @node_log / @step_log 埋点装饰器
    │   └── load_prompt.py       # 提示词加载与变量渲染
    ├── clients/                 # 外部存储客户端
    │   ├── milvus_utils.py      # 混合检索 / 按主键补全切片
    │   ├── minio_utils.py       # 图片上传与桶策略
    │   └── mongo_history_utils.py  # 会话历史读写
    ├── lm/                      # 模型封装（均为单例）
    │   ├── lm_utils.py          # ChatOpenAI 客户端（带缓存）
    │   ├── embedding_utils.py   # BGE-M3 双向量生成
    │   └── reranker_utils.py    # 精排模型
    ├── utils/                   # 通用工具
    │   ├── path_util.py         # 项目根目录定位
    │   ├── sse_utils.py         # SSE 队列与事件
    │   ├── task_utils.py        # 内存态任务进度追踪
    │   ├── milvus_hit_utils.py  # 检索结果拍平与去重
    │   ├── escape_milvus_string_utils.py  # 过滤表达式转义
    │   └── rate_limit_utils.py  # 第三方 API 滑动窗口限流
    ├── tool/                    # 模型下载脚本
    ├── import_process/          # 【导入服务】端口 8000
    │   ├── agent/
    │   │   ├── main_graph.py    # 流水线编排
    │   │   ├── state.py         # 状态定义
    │   │   └── nodes/           # 7 个处理节点
    │   ├── api/file_import_service.py
    │   └── page/import.html     # 上传页面
    └── query_process/           # 【查询服务】端口 8001
        ├── agent/
        │   ├── main_graph.py    # 流水线编排
        │   ├── state.py         # 状态定义
        │   ├── filters.py       # 条件 → Milvus 过滤表达式
        │   └── nodes/           # 6 个处理节点
        ├── api/query_service.py
        └── page/chat.html       # 对话页面
```

---

## 5. 快速开始

### 5.1 环境要求

| 依赖 | 版本 / 说明 |
| :--- | :--- |
| Python | ≥ 3.11 |
| Milvus | ≥ 2.4（查询侧用到 `like` 语法） |
| MongoDB | 用于会话历史 |
| MinIO | 用于图片对象存储 |
| 本地模型 | `BAAI/bge-m3`、`BAAI/bge-reranker-large` |
| 大模型 API | 通义千问（百炼）API Key |

模型缺失时可用脚本下载：

```bash
uv run python app/tool/download_bgem3.py
uv run python app/tool/download_reranker.py
```

### 5.2 安装依赖

```bash
uv sync
```

### 5.3 配置环境变量

```bash
cp .env.example .env          # Windows: Copy-Item .env.example .env
```

关键配置项：

| 变量 | 说明 |
| :--- | :--- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 大模型接口（OpenAI 兼容） |
| `LLM_DEFAULT_MODEL` / `VL_MODEL` | 对话模型 / 视觉模型 |
| `BGE_M3_PATH` | BGE-M3 本地模型路径 |
| `BGE_DEVICE` / `BGE_FP16` | 向量模型设备（`cpu` / `cuda:0`）与半精度 |
| `BGE_RERANKER_LARGE` / `BGE_RERANKER_DEVICE` | 精排模型路径与设备 |
| `MILVUS_URL` / `CHUNKS_COLLECTION` | Milvus 地址与切片集合名 |
| `MONGO_URL` / `MONGO_DB_NAME` | 会话历史存储 |
| `MINIO_*` | 图片对象存储 |
| `MINERU_BASE_URL` / `MINERU_API_TOKEN` | PDF 解析服务 |

> [!IMPORTANT]
> `CHUNKS_COLLECTION` 用于隔离不同知识库。同一 Milvus 实例上运行多个项目时，务必使用不同的集合名，否则会因 schema 不匹配导致插入失败或数据互相污染。

### 5.4 启动服务

导入服务与查询服务是**两个独立的进程**，需要**两个终端**分别启动。

```bash
# 终端 1：导入服务（8000）
uv run uvicorn app.import_process.api.file_import_service:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：查询服务（8001）
uv run uvicorn app.query_process.api.query_service:app --host 127.0.0.1 --port 8001 --reload
```

> [!TIP]
> 不要用 IDE 的「运行」按钮启动 —— 它会复用当前活动终端，导致第二个服务无法启动。请在新建的终端里手敲命令，或用 `Start-Process powershell` 弹独立窗口。

### 5.5 导入文档

1. 打开 `http://127.0.0.1:8000/import.html`
2. 拖入或选择 `.md` / `.pdf` 文件（支持一次多选批量上传）
3. 页面实时展示每个文件的处理进度与所在节点
4. 访问 `http://127.0.0.1:8000/stats` 查看向量库当前数据量

> [!NOTE]
> 首次导入会加载 BGE-M3 模型，第一篇约需 30~60 秒，属正常现象。CPU 环境下建议分批导入（每批 3~5 篇）。

### 5.6 开始问答

打开 `http://127.0.0.1:8001/chat.html`，输入育儿问题即可。

| 页面能力 | 说明 |
| :--- | :--- |
| 流程可视化 | 实时展示「解析条件 → 双路检索 → 融合 → 精排 → 生成」进度 |
| 流式输出 | 逐字返回，首字延迟低 |
| 消息操作 | 复制答案 / 重新生成 / 停止生成 |
| 历史管理 | 刷新自动恢复历史对话，支持一键清空会话 |

---

## 6. 接口说明

### 6.1 导入服务（:8000）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| `GET` | `/import.html` | 上传页面 |
| `POST` | `/upload` | 上传文件（`files` 字段，支持多文件），返回 `task_ids` |
| `GET` | `/status/{task_id}` | 查询任务进度（`status` / `running_list` / `done_list`） |
| `GET` | `/stats` | 查看向量库数据量 |

### 6.2 查询服务（:8001）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| `GET` | `/chat.html` | 对话页面 |
| `POST` | `/query` | 提问，入参 `{query, session_id, is_stream}` |
| `GET` | `/stream/{session_id}` | SSE 流式输出 |
| `GET` | `/history/{session_id}` | 查询最近历史对话 |
| `DELETE` | `/delete/{session_id}` | 清空会话历史 |
| `GET` | `/health` | 健康检查 |

**SSE 事件类型**

| 事件 | 载荷 | 说明 |
| :--- | :--- | :--- |
| `ready` | `{}` | 连接建立 |
| `progress` | `{status, done_list, running_list}` | 节点进度 |
| `delta` | `{delta}` | 答案增量片段 |
| `final` | `{answer}` | 完整答案 |
| `error` | `{error}` | 错误信息 |

---

## 7. 流水线设计

### 7.1 导入流水线

```mermaid
flowchart TD
    A["node_entry<br/>文件类型路由"] -->|".pdf"| B["node_pdf_to_md<br/>MinerU 解析"]
    A -->|".md"| C["node_md_img<br/>图片处理"]
    B --> C
    C --> D["node_document_split<br/>两段式切分"]
    D --> E["node_metadata_extract<br/>元数据抽取"]
    E --> F["node_bge_embedding<br/>双向量生成"]
    F --> G["node_import_milvus<br/>幂等入库"]
    G --> H(["END"])

    classDef main fill:#e7f0ff,stroke:#2f7aed,color:#1f3d5c
    classDef key fill:#e6f6ef,stroke:#22a06b,color:#1c5c40
    class A,B,C,D,F,G main
    class E key
```

| 节点 | 职责 |
| :--- | :--- |
| `node_entry` | 判断文件类型（PDF / MD），确定路由分支，提取文件标题与原始文件名 |
| `node_pdf_to_md` | 调用 MinerU 在线 API 将 PDF 解析为 Markdown |
| `node_md_img` | 扫描 md 内图片，调用视觉模型生成摘要，上传 MinIO 并替换链接；**无图时直接透传** |
| `node_document_split` | 按 Markdown 标题初切 + 超长段落二次切分，结果备份到 `backup.json` |
| `node_metadata_extract` | 正则解析 `## 元数据` 段，缺失时 LLM 兜底，并把元数据绑定到每个切片 |
| `node_bge_embedding` | 用 BGE-M3 为每个切片生成稠密 + 稀疏向量 |
| `node_import_milvus` | 按 schema 建集合，按来源文件删除旧数据（幂等），批量插入 |

### 7.2 查询流水线

```mermaid
flowchart TD
    A["node_filter_extract<br/>条件提取 + 问题改写"] -->|"已有答案<br/>（如空问题提示）"| G["node_answer_output<br/>生成答案"]
    A --> B["node_search_embedding<br/>混合检索"]
    A --> C["node_search_embedding_hyde<br/>HyDE 检索"]
    B --> D["node_rrf<br/>RRF 融合"]
    C --> D
    D --> E["node_rerank<br/>精排 Top-5"]
    E --> G
    G --> H(["END"])

    classDef main fill:#e7f0ff,stroke:#2f7aed,color:#1f3d5c
    classDef merge fill:#fff8e8,stroke:#e0932f,color:#7a5a14
    class A,B,C,E,G main
    class D merge
```

| 节点 | 职责 |
| :--- | :--- |
| `node_filter_extract` | 结合历史做指代消解，抽取检索条件并改写问题；落库用户消息 |
| `node_search_embedding` | 用改写后的问题做混合检索，带元数据过滤；无命中时自动退化为全库检索 |
| `node_search_embedding_hyde` | 先让 LLM 生成假设性答案，再据此检索，提升口语化提问的召回率 |
| `node_rrf` | 按 RRF（k=60）融合两路结果，按 `chunk_id` 去重 |
| `node_rerank` | bge-reranker-large 对候选逐对打分排序，输出 Top-5；失败时降级为 RRF 顺序 |
| `node_answer_output` | 组装带来源的上下文，生成四段式答案，流式推送并落库 |

### 7.3 关键参数

| 参数 | 取值 | 位置 |
| :--- | :--- | :--- |
| 切片长度 / 重叠 | `500 / 50` | `node_document_split.py` |
| 向量化批大小 | `5` | `node_bge_embedding.py` |
| 混合检索权重（稠密, 稀疏） | `(0.8, 0.2)` | `node_search_embedding*.py` |
| 单路召回 / 融合返回 | `10 / 5` | 同上 |
| RRF 平滑常数 k | `60` | `node_rrf.py` |
| 精排候选 / 最终 Top-K | `20 / 5` | `node_rerank.py` |

---

## 8. 数据规范

知识库中每篇文档建议在头部包含 `## 元数据` 段：

```markdown
# 3-6岁情绪管理与规则建立建议
## 元数据
- 内容类型：专家建议
- 年龄段：3-6岁
- 问题类型：情绪管理 / 行为引导 / 亲子沟通
- 场景描述：发脾气、不肯合作、规则执行困难、公共场所失控

## 核心观点
...
```

字段约束（定义在 `app/conf/domain_config.py`）：

| 字段 | 中文名 | 取值 |
| :--- | :--- | :--- |
| `content_type` | 内容类型 | 育儿建议 / 专家建议 / 亲子案例 / 沟通话术 / 知识科普 |
| `age_range` | 年龄段 | 0-3岁 / 3-6岁 / 6-12岁 / 12+岁（可多值） |
| `problem_type` | 问题类型 | 情绪管理 / 行为引导 / 学习能力 / 社交能力 / 亲子沟通 等（可多值） |
| `scene` | 场景描述 | 自由文本，多值以顿号或斜杠分隔 |

<details>
<summary><b>多值字段的处理规则（点击展开）</b></summary>

多值字段在入库前会被规范化为**英文逗号拼接**：

```
"3-6岁 / 6-12岁"  →  "3-6岁,6-12岁"
```

检索时用 `like "%3-6岁%"` 做包含匹配，同字段内多值取 `or`、不同字段间取 `and`：

```sql
(age_range like "%3-6岁%" or age_range like "%6-12岁%")
  and problem_type like "%情绪管理%"
```

</details>

---

## 9. 向量库 Schema

<details>
<summary><b>集合 kb_parenting_chunks 的完整字段定义（点击展开）</b></summary>

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `chunk_id` | `INT64` | 主键（自增） |
| `file_title` | `VARCHAR` | 来源文件名（去后缀） |
| `source_file` | `VARCHAR` | 来源文件名（含后缀），幂等删除依据 |
| `source_path` | `VARCHAR` | 来源路径 |
| `content_type` | `VARCHAR` | 内容类型 |
| `age_range` | `VARCHAR` | 年龄段（逗号拼接） |
| `problem_type` | `VARCHAR` | 问题类型（逗号拼接） |
| `scene` | `VARCHAR` | 场景描述（逗号拼接） |
| `title` | `VARCHAR` | 切片标题 |
| `parent_title` | `VARCHAR` | 所属章节标题 |
| `part` | `INT8` | 切片在同章节内的序号 |
| `author` | `VARCHAR` | 作者 |
| `content` | `VARCHAR` | 切片正文 |
| `dense_vector` | `FLOAT_VECTOR(1024)` | 稠密向量 · HNSW / COSINE（M=32, efConstruction=300） |
| `sparse_vector` | `SPARSE_FLOAT_VECTOR` | 稀疏向量 · SPARSE_INVERTED_INDEX / IP |

</details>

> 所有标量字段名统一由 `app/conf/domain_config.py` 的 `CHUNK_SCALAR_FIELDS` 定义，建表与插入共用同一份列表，从根本上避免两侧字段名不一致。

---

## 10. 关键设计说明

### 10.1 元数据"正则优先 + LLM 兜底"

素材头部已带规范的元数据段，正则解析**准确率 100% 且零成本**；仅在完全解析不到时才调用大模型。把成本花在真正需要的文档上。

### 10.2 向量化文本注入元数据

切片向量化时使用如下模板作为输入：

```
【内容类型】【年龄段】切片标题：正文
```

让「3-6岁」「情绪管理」这类条件词也参与语义召回，而不仅仅依赖正文相似度。

### 10.3 两级召回兜底

带条件检索无命中时，**自动去掉过滤条件重查一次**，避免"条件写得太细导致答不出来"。

### 10.4 RRF 融合

只看排名不看绝对分数，规避两路检索器分数量纲不可比的问题：

```
score(d) = Σ 1 / (k + rank_i(d))        k = 60
```

### 10.5 全链路降级

| 环节 | 失败时 |
| :--- | :--- |
| HyDE 文档生成 | 退化为普通向量检索 |
| 精排模型 | 降级使用 RRF 排序结果 |
| 流式生成中断 | 降级为一次性生成 |
| 元数据正则解析 | 调用 LLM 兜底 |

### 10.6 幂等导入

同一份文档重复导入时，先按 `source_file` 删除旧切片再插入，保证是"替换"而非"追加"。

---

## 11. 常见问题

<details>
<summary><b>Q：为什么导入服务与查询服务要分两个端口？</b></summary>

两条链路的资源占用与生命周期不同：导入是重计算任务（向量化耗时），查询是低延迟任务。拆成两个进程后可以独立重启、独立扩容，导入服务挂了也不影响线上问答。

</details>

<details>
<summary><b>Q：启动报 <code>FileNotFoundError: 未找到项目根目录标识「.env」</code>？</b></summary>

`app/utils/path_util.py` 以 `.env` 作为项目根目录的标识文件。请确认 `.env` 已创建在项目根目录（从 `.env.example` 复制）。

</details>

<details>
<summary><b>Q：导入时日志停在"开始为 N 条文本生成混合向量嵌入"很久不动？</b></summary>

BGE-M3 正在 CPU 上编码，属正常现象（该步骤内部无中间日志）。可打开任务管理器确认 python 进程的 CPU 占用：

- CPU 占用高 → 正在计算，继续等待
- CPU 接近 0 且内存不动 → 检查是否内存不足（BGE-M3 fp32 约占 2~3 GB）

启用 GPU 可显著加速：`.env` 中设置 `BGE_DEVICE=cuda:0`、`BGE_FP16=1`。

</details>

<details>
<summary><b>Q：查询总是返回"知识库中暂未收录相关内容"？</b></summary>

按顺序排查：

1. 访问 `http://127.0.0.1:8000/stats`，确认 `row_count > 0`（数据是否真的入库）
2. 查看日志中 `node_rerank` 输出的 Top-5 命中，判断是检索质量问题还是知识库确实没有该内容
3. 检查 `.env` 的 `CHUNKS_COLLECTION` 与导入时使用的是否一致

</details>

<details>
<summary><b>Q：流式接口无输出或报错？</b></summary>

`POST /query` 必须带 `is_stream: true`，服务端才会创建 SSE 队列；随后前端通过 `GET /stream/{session_id}` 建立连接。若用 `/docs` 手动测试，注意两者要配合使用。

</details>

<details>
<summary><b>Q：同一个 Milvus 实例上能跑多个知识库吗？</b></summary>

可以。通过 `.env` 的 `CHUNKS_COLLECTION` 指定不同集合名即可隔离。**不要复用已有集合**——若同名集合已存在，代码会跳过建表，导致新旧 schema 不匹配。

</details>

---

## 12. 后续扩展方向

- 内容后台管理与知识库管理页面
- 文档增量更新（按变更检测触发重导入）
- 权限控制与多租户支持
- 个性化推荐与亲子画像
- 情感识别与高风险信号预警
- 对话结果返回知识卡片结构（前端卡片化渲染）

---

<div align="center">

**育儿知识库** · 基于 FastAPI + LangGraph + Milvus 的 RAG 实践

</div>
