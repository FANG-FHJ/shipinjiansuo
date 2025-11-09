# src/search.py
import os

# 强制设置镜像源 - 在导入任何其他库之前
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


class VideoRetrievalSystem:
    def __init__(self, database_dir, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.database_dir = database_dir

        # 加载特征库
        print("Loading feature database...")
        with open(f"{database_dir}/feature_database.json", 'r', encoding='utf-8') as f:
            self.feature_db = json.load(f)
        with open(f"{database_dir}/video_list.json", 'r', encoding='utf-8') as f:
            self.video_list = json.load(f)

        # 转换为NumPy矩阵
        self.feature_matrix = np.array([self.feature_db[name] for name in self.video_list])

        # 加载CLIP模型
        print(f"Loading CLIP model: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print("System initialized successfully!")

    def search(self, query_text, top_k=3):
        """根据文本查询搜索视频"""
        # 文本编码
        with torch.no_grad():
            text_inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_feature = self.model.get_text_features(**text_inputs).cpu().numpy()

        # 计算相似度
        similarities = cosine_similarity(text_feature, self.feature_matrix).flatten()

        # 获取Top-K结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for rank, idx in enumerate(top_indices):
            video_name = self.video_list[idx]
            score = similarities[idx]
            results.append({
                "rank": rank + 1,
                "video": video_name,
                "score": round(score, 4)
            })

        return results


def run_complete_experiment():
    """运行完整实验"""
    # 初始化系统
    system = VideoRetrievalSystem(database_dir="../data/database")

    # 定义测试集
    test_queries = [
        # 基础物体
        "person", "car", "animal", "tree", "building",
        # 场景
        "outdoor", "indoor", "city", "nature", "urban",
        # 动作
        "walking", "running", "moving", "standing",
        # 属性
        "bright", "dark", "colorful", "large",
        # 复杂查询
        "person walking", "car moving", "outdoor scene"
    ]

    print("开始视频检索系统实验验证")
    print(f"测试查询数量: {len(test_queries)}")
    print(f"视频数据库大小: {len(system.video_list)} 个视频")

    # 执行批量测试
    results = {}
    for query in test_queries:
        results[query] = system.search(query, top_k=3)
        print(f"查询: '{query}'")
        for res in results[query]:
            print(f"  {res['rank']}. {res['video']} (相似度: {res['score']:.4f})")

    # 分析结果
    analyze_experiment_results(results, system.video_list)

    return results


def analyze_experiment_results(results, video_list):
    """分析实验结果的函数"""
    print("\n" + "=" * 50)
    print("实验结果分析")
    print("=" * 50)

    # 1. 统计相似度分布
    all_scores = []
    for query, result_list in results.items():
        for res in result_list:
            all_scores.append(res['score'])

    print(f"相似度统计:")
    print(f"  平均相似度: {np.mean(all_scores):.4f}")
    print(f"  最高相似度: {np.max(all_scores):.4f}")
    print(f"  最低相似度: {np.min(all_scores):.4f}")
    print(f"  相似度标准差: {np.std(all_scores):.4f}")

    # 2. 分析查询类型效果
    query_categories = {
        "物体查询": ["person", "car", "animal", "tree", "building"],
        "场景查询": ["outdoor", "indoor", "city", "nature", "urban"],
        "动作查询": ["walking", "running", "moving", "standing"]
    }

    for category, queries in query_categories.items():
        category_scores = []
        for query in queries:
            if query in results:
                for res in results[query]:
                    category_scores.append(res['score'])
        if category_scores:
            print(f"\n{category}效果:")
            print(f"  平均相似度: {np.mean(category_scores):.4f}")
            print(f"  测试数量: {len(category_scores)}")


def main():
    # 初始化系统
    system = VideoRetrievalSystem(database_dir="../data/database")

    # 交互式搜索
    print("\n" + "=" * 50)
    print("视频检索系统已启动！")
    print("输入查询文本（或输入 'quit' 退出）：")

    while True:
        query = input("\n查询: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        # 执行搜索
        results = system.search(query, top_k=3)

        # 显示结果
        print(f"\n'{query}' 的搜索结果：")
        for res in results:
            print(f"  {res['rank']}. {res['video']} (相似度: {res['score']})")


# 修改文件末尾的代码
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        run_complete_experiment()
    else:
        main()  # 原来的交互模式