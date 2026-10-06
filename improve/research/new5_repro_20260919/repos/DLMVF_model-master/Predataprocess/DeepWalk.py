import numpy as np
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

# Step 1: 构建图结构
def build_graph(association_matrix):
    """
    association_matrix: miRNA 与基因的关联矩阵 (numpy.ndarray)
    返回构建的无向图 (networkx.Graph)
    """
    n_miRNA, n_genes = association_matrix.shape
    G = nx.Graph()

    # 添加 miRNA 节点
    for i in range(n_miRNA):
        G.add_node(f"miRNA_{i}")

    # 添加基因节点
    for j in range(n_genes):
        G.add_node(f"Gene_{j}")

    # 添加边
    for i in range(n_miRNA):
        for j in range(n_genes):
            if association_matrix[i, j] > 0:  # 有关联
                G.add_edge(f"miRNA_{i}", f"Gene_{j}", weight=association_matrix[i, j])

    return G

# Step 2: 实现随机游走
def random_walk(graph, start_node, walk_length):
    """
    在图中进行随机游走
    graph: networkx.Graph
    start_node: 起始节点
    walk_length: 游走步长
    返回节点序列
    """
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if not neighbors:  # 没有邻居则停止
            break
        next_node = np.random.choice(neighbors)
        walk.append(next_node)
    return walk

# Step 3: 生成随机游走序列
def generate_walks(graph, num_walks, walk_length):
    """
    生成随机游走序列
    graph: networkx.Graph
    num_walks: 每个节点的随机游走次数
    walk_length: 每次游走的步长
    返回随机游走序列 (list of list)
    """
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)  # 随机打乱节点顺序
        for node in nodes:
            walks.append(random_walk(graph, node, walk_length))
    return walks

# Step 4: 使用 Word2Vec 训练嵌入
def train_deepwalk(walks, embedding_size, window_size, epochs):
    """
    使用 Word2Vec 训练 DeepWalk 嵌入
    walks: 随机游走序列
    embedding_size: 嵌入维度
    window_size: Word2Vec 的窗口大小
    epochs: 训练轮数
    返回节点嵌入表示 (gensim.Word2Vec)
    """
    walks = [[str(node) for node in walk] for walk in walks]  # 转换为字符串
    model = Word2Vec(
        walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=0,
        sg=1,  # Skip-gram 模型
        workers=4,
        epochs=epochs,
    )
    return model

# Step 5: 运行 DeepWalk
def deepwalk_pipeline(association_matrix, num_walks=10, walk_length=80, embedding_size=512, window_size=5, epochs=10):
    """
    DeepWalk 管道
    association_matrix: miRNA 与基因的关联矩阵
    num_walks: 每个节点的随机游走次数
    walk_length: 每次游走的步长
    embedding_size: 嵌入维度
    window_size: Word2Vec 的窗口大小
    epochs: 训练轮数
    返回节点嵌入字典 {节点: 嵌入向量}
    """

    # 构建图
    graph = build_graph(association_matrix)
    print("Graph constructed.")

    # 生成随机游走序列
    walks = generate_walks(graph, num_walks, walk_length)
    print("Random walks generated.")

    # 训练嵌入
    model = train_deepwalk(walks, embedding_size, window_size, epochs)
    print("DeepWalk model trained.")

    # 返回节点嵌入
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return embeddings
def save_embeddings_to_txt(embeddings, output_file, prefix="miRNA"):
    """
    保存 miRNA 节点的嵌入到一个文本文件
    embeddings: 节点嵌入字典 {节点: 嵌入向量}
    output_file: 输出的文件路径
    prefix: 过滤的节点前缀（默认是 miRNA 节点）
    """
    with open(output_file, "w") as f:
        for node, embedding in embeddings.items():
            if node.startswith(prefix):  # 只保存以 prefix 开头的节点
                embedding_str = "\t".join(map(str, embedding))
                f.write(f"{embedding_str}\n")
    print(f"Embeddings saved to {output_file}.")


df = pd.read_excel('../data/association_matrix.xlsx', index_col=0)
association_matrix=df.to_numpy()
A=df.to_numpy()
embeddings = deepwalk_pipeline(association_matrix, num_walks=10, walk_length=10, embedding_size=512, window_size=3, l=5)
save_embeddings_to_txt(embeddings, "miRNA_embeddings.txt", prefix="miRNA")
