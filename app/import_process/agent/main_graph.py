"""
导入流水线 LangGraph 编排

流程：
node_entry ──(条件：pdf / md)──> node_pdf_to_md ─> node_md_img ─> node_document_split
        └────────────────────────> node_md_img ─┘
                                            └─> node_metadata_extract ─> node_bge_embedding
                                                                              └─> node_import_milvus ─> END
"""
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_metadata_extract import node_metadata_extract
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState

# 初始化环境变量
load_dotenv()

# 构建图
workflow = StateGraph(ImportGraphState)

# 注册节点
workflow.add_node(node_entry)
workflow.add_node(node_pdf_to_md)
workflow.add_node(node_md_img)
workflow.add_node(node_document_split)
workflow.add_node(node_metadata_extract)
workflow.add_node(node_bge_embedding)
workflow.add_node(node_import_milvus)


def condition_fun(state: ImportGraphState):
    """根据文件类型决定走 PDF 解析还是直接处理 md"""
    if state["is_md_read_enabled"]:
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    else:
        # 既不是 pdf 也不是 md：系统不支持，直接结束
        return END


# 入口
workflow.set_entry_point("node_entry")
workflow.add_conditional_edges(
    "node_entry",
    condition_fun,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)

# 顺序边
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_metadata_extract")
workflow.add_edge("node_metadata_extract", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

kb_import_app = workflow.compile()
