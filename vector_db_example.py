import numpy as np
import faiss

# 生成 10000 个 128 维的随机向量
d = 128
nb = 10000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype("float32")

# 构建索引

index = faiss.IndexFlatL2(d)  # L2 距离
index.add(xb)  # 向量入库
# 查询
xq = np.random.random((5, d)).astype("float32")  # 5 个查询向量
D, I = index.search(xq, k=3)  # 返回每个查询向量最近的3个向量
print("最近邻索引：", I)
print("距离：", D)


# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# # 连接 Milvus
# connections.connect("default", host="localhost", port="19530")

# # 定义 schema
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
# ]
# schema = CollectionSchema(fields, "向量集合示例")
# collection = Collection("demo_collection", schema)

# # 插入数据
# import numpy as np
# vectors = np.random.random((10, 128)).tolist()
# collection.insert([vectors])

# # 创建索引
# collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# # 查询
# search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
# results = collection.search(vectors[:2], "embedding", search_params, limit=3)
# print(results)
