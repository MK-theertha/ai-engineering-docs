# 08. Vector Databases

[Home](README.md) › 08. Vector Databases

> **Level:** Intermediate · **Time:** ~4 hours · **Prerequisites:** [Chapter 06](06-embeddings.md)

## Learning objectives

- Explain what a vector database is and why a regular database isn't enough.
- Understand exact vs approximate nearest-neighbour (ANN) search.
- Explain how HNSW, IVF and product quantization indexes work.
- Use ChromaDB, FAISS, Pinecone, Weaviate and pgvector from Python.
- Choose the right vector store for a project.

---

## 1. What are vector databases?

A **vector database** stores **embeddings** together with their source data and **metadata**, and finds the vectors **most similar** to a query vector very quickly, even across billions of items.

| Traditional database | Vector database |
|---|---|
| Exact matches: `WHERE email = 'a@b.com'` | Similarity: "find the 10 closest vectors to this one" |
| B-tree / hash indexes | ANN indexes (HNSW, IVF, PQ) |
| Results are correct or not | Results are ranked by distance |
| Structured data | Embeddings + metadata + original text |

### Why they matter

A brute-force comparison of a query against 10 million 1536-dimensional vectors means ~15 billion multiplications per query. Vector databases use **approximate nearest-neighbour (ANN)** indexes to answer in milliseconds, with 95–99%+ of the accuracy of an exact search.

They also handle production needs: persistence, CRUD updates, metadata filtering, hybrid search, multi-tenancy, replication and scaling.

```mermaid
flowchart LR
    D["Documents"] --> E["Embedding model"]
    E --> V["Vectors + metadata"]
    V --> IDX[("Vector DB: ANN index")]
    Q["Query"] --> QE["Embedding model"]
    QE --> S["Similarity search + metadata filter"]
    IDX --> S
    S --> R["Top-k results with scores"]
```

---

## 2. Similarity search

### Exact (k-NN) search

Compare the query with **every** vector. Perfect recall, but O(N·d) per query. Fine for up to roughly 100k vectors.

```python
import numpy as np

def knn_search(query: np.ndarray, vectors: np.ndarray, k: int = 5):
    sims = vectors @ query                    # vectors pre-normalised -> cosine
    top = np.argpartition(-sims, k)[:k]       # O(N), faster than a full sort
    return top[np.argsort(-sims[top])], sims

rng = np.random.default_rng(0)
vectors = rng.normal(size=(100_000, 384)).astype("float32")
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
query = vectors[42] + 0.01 * rng.normal(size=384)
query /= np.linalg.norm(query)

idx, sims = knn_search(query, vectors)
print(idx[:5], sims[idx][:5])   # 42 should be first
```

### Approximate nearest-neighbour (ANN) search

ANN trades a little accuracy (**recall**) for large gains in **speed** and **memory**.

| Index | How it works | Strengths | Weaknesses |
|---|---|---|---|
| **Flat** | Brute force | 100% recall | Slow at scale |
| **HNSW** (Hierarchical Navigable Small World) | Multi-layer graph; greedy navigation from coarse to fine layers | Fast, high recall; the most popular | High memory; slower to build |
| **IVF** (Inverted File) | Cluster vectors with k-means; search only the closest `nprobe` clusters | Lower memory, fast | Needs training; recall depends on `nprobe` |
| **PQ** (Product Quantization) | Split vectors into sub-vectors and compress each to a code | 10–100× compression | Lossy, lower recall |
| **IVF-PQ** | IVF + PQ | Billion-scale on limited RAM | Tuning complexity |
| **DiskANN** | Graph index on SSD | Huge indexes with little RAM | Disk I/O latency |

### How HNSW works

```mermaid
flowchart TB
    subgraph L2["Layer 2: few nodes, long jumps"]
        a2(("A")) --- d2(("D"))
    end
    subgraph L1["Layer 1: more nodes"]
        a1(("A")) --- b1(("B")) --- d1(("D")) --- f1(("F"))
    end
    subgraph L0["Layer 0: all nodes, short links"]
        a0(("A")) --- b0(("B")) --- c0(("C")) --- d0(("D")) --- e0(("E")) --- f0(("F")) --- g0(("G"))
    end
    a2 -.-> a1
    d2 -.-> d1
    a1 -.-> a0
    d1 -.-> d0
```

1. Each vector is a node, inserted into layer 0 and, with decreasing probability, into higher layers.
2. A search starts at the top layer and **greedily** moves to the neighbour closest to the query.
3. When no neighbour is closer, it drops a layer and continues with finer links.
4. On layer 0 it explores a candidate list of size `ef_search` and returns the best k.

Key parameters:

| Parameter | Meaning | Trade-off |
|---|---|---|
| `M` | Links per node | Higher → better recall, more memory |
| `ef_construction` | Candidate list size during build | Higher → better graph, slower build |
| `ef_search` | Candidate list size during search | Higher → better recall, slower queries |

### Metadata filtering

Real queries combine similarity with filters: *"similar chunks **where** department = 'HR' **and** year ≥ 2025"*.

- **Pre-filtering:** filter first, then search. Accurate, but can be slow or break the ANN graph.
- **Post-filtering:** search first, then filter. Fast, but can return fewer than k results.
- Modern databases use **filtered/integrated** strategies that handle both well.

---

## 3. Cosine similarity, distance metrics and gotchas

| Metric | Database setting names | Use when |
|---|---|---|
| Cosine | `cosine` | Text embeddings (the default choice) |
| Dot / inner product | `ip`, `dotproduct` | Normalised embeddings (same ranking as cosine, faster) |
| Euclidean (L2) | `l2`, `euclidean` | Models trained with L2; some image embeddings |

Some databases return **distance** (lower = better) and some **similarity** (higher = better). For cosine: `distance = 1 − similarity`. Always check which one you get.

> **Rule:** use the metric the embedding model was trained for. Most text embedding models are designed for cosine; OpenAI embeddings are normalised, so cosine, dot product and L2 give the same ranking.

---

## 4. ChromaDB

**Chroma** is an open-source, developer-friendly vector database. It's great for prototypes and small-to-medium apps, and runs in-process, as a local server, or as Chroma Cloud.

```bash
pip install chromadb
```

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_store")

# Chroma can embed for you (default: a small local model), or you can pass your own vectors
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name="products",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

collection.upsert(
    ids=["p1", "p2", "p3", "p4"],
    documents=[
        "Wireless noise-cancelling headphones with 30-hour battery",
        "Running shoes with breathable mesh and cushioned sole",
        "Bluetooth earbuds, waterproof, ideal for workouts",
        "Leather office chair with lumbar support",
    ],
    metadatas=[
        {"category": "audio", "price": 199},
        {"category": "sports", "price": 89},
        {"category": "audio", "price": 59},
        {"category": "furniture", "price": 249},
    ],
)

results = collection.query(
    query_texts=["earphones for the gym"],
    n_results=2,
    where={"$and": [{"category": "audio"}, {"price": {"$lt": 100}}]},
)
print(results["documents"], results["distances"])

collection.update(ids=["p3"], metadatas=[{"category": "audio", "price": 49}])
collection.delete(ids=["p4"])
print(collection.count())
```

**Best for:** learning, prototypes, local RAG, small production apps.

---

## 5. FAISS

**FAISS** (Facebook AI Similarity Search, from Meta) is a **library**, not a database. It provides extremely fast in-memory vector indexes on CPU and GPU. You manage persistence, metadata and IDs yourself.

```bash
pip install faiss-cpu
```

```python
import faiss
import numpy as np

d = 384
rng = np.random.default_rng(0)
xb = rng.normal(size=(200_000, d)).astype("float32")
faiss.normalize_L2(xb)                         # normalise -> inner product = cosine

# 1) Exact search
flat = faiss.IndexFlatIP(d)
flat.add(xb)

# 2) HNSW (approximate, fast, no training)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)   # M = 32
hnsw.hnsw.efSearch = 64
hnsw.add(xb)

# 3) IVF-PQ (compressed, needs training)
nlist, m = 1024, 48                            # clusters, sub-quantizers (d must be divisible by m)
quantizer = faiss.IndexFlatIP(d)
ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
ivfpq.train(xb)
ivfpq.add(xb)
ivfpq.nprobe = 16

xq = xb[:5].copy()
for name, index in [("flat", flat), ("hnsw", hnsw), ("ivfpq", ivfpq)]:
    scores, ids = index.search(xq, 5)
    print(name, ids[0])

faiss.write_index(hnsw, "products.hnsw")       # persist to disk
index = faiss.read_index("products.hnsw")
```

Keep metadata in a separate store (a dict, SQLite or Parquet) keyed by the FAISS row ID, or use `IndexIDMap` to attach your own IDs.

**Best for:** maximum speed, research, offline batch similarity, large in-memory indexes, GPU search.

---

## 6. Pinecone

**Pinecone** is a fully managed, serverless vector database. There's no infrastructure to run, and it scales automatically.

```bash
pip install pinecone
```

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX = "docs-index"

if not pc.has_index(INDEX):
    pc.create_index(
        name=INDEX,
        dimension=1536,                   # must match your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX)

# vectors from e.g. OpenAI text-embedding-3-small
index.upsert(
    vectors=[
        {"id": "doc1#0", "values": [0.01] * 1536, "metadata": {"source": "hr.pdf", "dept": "HR"}},
        {"id": "doc2#0", "values": [0.02] * 1536, "metadata": {"source": "it.pdf", "dept": "IT"}},
    ],
    namespace="tenant-acme",              # namespaces isolate tenants
)

res = index.query(
    vector=[0.01] * 1536,
    top_k=3,
    include_metadata=True,
    filter={"dept": {"$eq": "HR"}},
    namespace="tenant-acme",
)
for match in res.matches:
    print(match.id, match.score, match.metadata)
```

**Best for:** teams that want managed infrastructure, fast time-to-production, serverless scaling and multi-tenancy with namespaces.

---

## 7. Weaviate

**Weaviate** is an open-source vector database (self-hosted or Weaviate Cloud) with built-in **hybrid search**, optional built-in vectorisers, multi-tenancy and a GraphQL/gRPC API.

```bash
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
pip install weaviate-client
```

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = weaviate.connect_to_local()

try:
    if client.collections.exists("Article"):
        client.collections.delete("Article")

    articles = client.collections.create(
        name="Article",
        vector_config=Configure.Vectors.self_provided(),     # we supply our own vectors
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="body", data_type=DataType.TEXT),
        ],
    )

    data = [
        ("Reset password", "Click 'Forgot password' on the login page to reset it."),
        ("Refund policy", "Refunds are issued within 7 days of receiving the item."),
    ]
    with articles.batch.fixed_size(batch_size=100) as batch:
        for title, body in data:
            batch.add_object(properties={"title": title, "body": body},
                             vector=model.encode(body).tolist())

    query = "I can't log in"
    res = articles.query.hybrid(               # BM25 + vector in one call
        query=query,
        vector=model.encode(query).tolist(),
        alpha=0.7,                             # 1 = pure vector, 0 = pure keyword
        limit=2,
        return_metadata=MetadataQuery(score=True),
    )
    for obj in res.objects:
        print(obj.properties["title"], obj.metadata.score)
finally:
    client.close()
```

> The Weaviate Python client v4 API changes between minor versions (for example `vector_config` vs the older `vectorizer_config`). Check the [client docs](https://docs.weaviate.io/weaviate/client-libraries/python) for your installed version.

**Best for:** native hybrid search, self-hosting with rich features, multi-tenant SaaS.

---

## 8. pgvector

**pgvector** is a PostgreSQL extension that adds a `vector` type, distance operators and ANN indexes (HNSW, IVFFlat). It keeps vectors **next to your relational data**, so you get joins, transactions, backups and SQL filters for free.

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg17
pip install "psycopg[binary]" pgvector
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    department  TEXT,
    content     TEXT NOT NULL,
    embedding   VECTOR(384)
);

-- HNSW index using cosine distance
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

| Operator | Distance |
|---|---|
| `<->` | L2 (Euclidean) |
| `<=>` | Cosine distance |
| `<#>` | Negative inner product |

```python
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/postgres", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id BIGSERIAL PRIMARY KEY, source TEXT, department TEXT,
        content TEXT, embedding VECTOR(384))
""")
conn.execute("CREATE INDEX IF NOT EXISTS docs_hnsw ON documents USING hnsw (embedding vector_cosine_ops)")

rows = [
    ("hr.pdf", "HR", "Employees get 24 days of paid annual leave."),
    ("it.pdf", "IT", "VPN access requires two-factor authentication."),
]
with conn.cursor() as cur:
    for source, dept, text in rows:
        cur.execute(
            "INSERT INTO documents (source, department, content, embedding) VALUES (%s, %s, %s, %s)",
            (source, dept, text, model.encode(text)),
        )

query_vec = model.encode("How many holidays do I get?")
results = conn.execute(
    """
    SELECT source, content, 1 - (embedding <=> %s) AS similarity
    FROM documents
    WHERE department = %s
    ORDER BY embedding <=> %s
    LIMIT 3
    """,
    (query_vec, "HR", query_vec),
).fetchall()
for r in results:
    print(r)
```

**Best for:** teams already on PostgreSQL, apps that need joins and transactions, and moderate scale (millions of vectors) without extra infrastructure.

---

## 9. Comparison and how to choose

| | ChromaDB | FAISS | Pinecone | Weaviate | pgvector |
|---|---|---|---|---|---|
| Type | Open-source DB | Library | Managed SaaS | Open-source DB + cloud | Postgres extension |
| Hosting | Embedded / server / cloud | In-process | Fully managed | Self-host / cloud | Wherever Postgres runs |
| Persistence | Yes | Manual (save/load) | Yes | Yes | Yes |
| Metadata filtering | Yes | Manual | Yes | Yes | Full SQL |
| Hybrid search | Basic/full-text depending on deployment | No | Yes (sparse-dense) | Native | With Postgres full-text search |
| Scale | Small–medium | Very large (in RAM/GPU) | Very large | Large | Medium–large |
| Ops effort | Very low | Low (but DIY features) | None | Medium | Low if you already run Postgres |
| Best for | Learning, prototypes | Speed, research | Managed production | Hybrid search, multi-tenant | Existing Postgres apps |

Other popular options: **Qdrant** (Rust, strong filtering), **Milvus/Zilliz** (billion-scale), **Elasticsearch/OpenSearch** (keyword + vector), **MongoDB Atlas Vector Search**, **Redis**, **LanceDB**.

```mermaid
flowchart TD
    A{"Already using PostgreSQL?"} -->|"Yes, under 10M vectors"| PG["pgvector"]
    A -->|No| B{"Want zero ops?"}
    B -->|Yes| PC["Pinecone or another managed service"]
    B -->|No| C{"Prototype or learning?"}
    C -->|Yes| CH["ChromaDB"]
    C -->|No| D{"Need native hybrid search or multi-tenancy?"}
    D -->|Yes| WV["Weaviate / Qdrant"]
    D -->|"No, need raw in-process speed"| FA["FAISS"]
```

---

## 10. Practical example: swappable vector store interface

Write your RAG code against an interface so you can switch stores without rewriting the app.

```python
from typing import Protocol
import chromadb

class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]], texts: list[str], metadatas: list[dict]) -> None: ...
    def search(self, vector: list[float], k: int = 5, filters: dict | None = None) -> list[dict]: ...

class ChromaStore:
    def __init__(self, name: str, path: str = "./store"):
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, vectors, texts, metadatas):
        self.col.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    def search(self, vector, k=5, filters=None):
        r = self.col.query(query_embeddings=[vector], n_results=k, where=filters)
        return [{"id": i, "text": t, "metadata": m, "score": 1 - d}
                for i, t, m, d in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

# A PgVectorStore or PineconeStore class can implement the same two methods.
```

---

## 11. Real-world use cases

| Use case | Typical store |
|---|---|
| Startup RAG MVP | Chroma or pgvector |
| SaaS product with per-customer knowledge bases | Pinecone namespaces, Weaviate multi-tenancy, Qdrant |
| E-commerce recommendations over 100M items | Milvus, FAISS on GPU, managed services |
| Enterprise search mixing keyword + semantic | Weaviate, Elasticsearch/OpenSearch |
| App with existing Postgres and user tables | pgvector (join vectors with permissions) |

---

## 12. Advantages, disadvantages and limitations

| Advantages | Disadvantages / limitations |
|---|---|
| Millisecond semantic search at scale | ANN is approximate: recall < 100% |
| Metadata filtering + similarity in one query | Memory-hungry (HNSW keeps graphs in RAM) |
| Managed options remove ops work | Another system to run, secure and back up |
| Hybrid search improves exact-term recall | Index rebuilds needed when you change embedding models |
| | Filtering + ANN can reduce recall if poorly implemented |
| | Costs grow with dimensions × vectors × replicas |

---

## 13. Best practices and common mistakes

### Best practices

- Store **text + metadata + embedding model name/version** with every vector.
- Match the index **metric** to the embedding model.
- Use **deterministic IDs** (content hashes) and **upserts** for idempotent indexing.
- **Benchmark recall vs latency**: compare ANN results against a flat index on a sample.
- Use **namespaces/tenants/filters** for data isolation; enforce them server-side.
- **Batch** inserts; build indexes after bulk loads when possible.
- Plan for **re-embedding** (blue/green indexes) when you upgrade models.

### Common mistakes

| Mistake | Fix |
|---|---|
| Dimension mismatch between model and index | Create the index with the model's exact dimension |
| Treating distance as similarity | Check the docs; convert (`1 - distance` for cosine) |
| Storing only vectors, without text | Store the text or a reliable pointer to it |
| Using FAISS and expecting DB features | Use a database if you need filters, CRUD and persistence |
| Default HNSW params for every workload | Tune `ef_search` / `M` against recall targets |
| Multi-tenant data in one unfiltered index | Namespaces or mandatory tenant filters |

---

## 14. Interview questions and answers

**Q1. Why can't we just use a regular database for embeddings?**
B-tree and hash indexes are built for exact matches and ranges, not high-dimensional similarity. Brute-force similarity over millions of vectors is too slow. Vector databases provide ANN indexes plus filtering, persistence and scaling designed for this workload. (Postgres with pgvector shows that a regular database *can* do it once you add such an index.)

**Q2. Explain HNSW.**
A multi-layer proximity graph. Upper layers have few nodes with long-range links, and lower layers have all nodes with short-range links. Search starts at the top, greedily moves towards the query, then descends layer by layer, giving roughly logarithmic search time with high recall. `M`, `ef_construction` and `ef_search` trade memory and speed against recall.

**Q3. What's the difference between IVF and HNSW?**
IVF clusters vectors and searches only the nearest clusters (`nprobe`). It uses less memory and pairs well with PQ compression, but needs training. HNSW is graph-based, needs no training, usually gives better recall/latency, and uses more memory.

**Q4. What is product quantization?**
A compression technique: split each vector into m sub-vectors and replace each sub-vector with the ID of its nearest centroid in a learned codebook. A 1536-float vector can shrink to tens of bytes, and distances are approximated using lookup tables.

**Q5. How do you choose between Pinecone, pgvector and Chroma?**
Chroma for prototypes and small apps. pgvector when you already run Postgres and want vectors next to relational data with SQL filters and transactions. Pinecone (or another managed service) when you want serverless scale without ops. Consider scale, hybrid search needs, multi-tenancy, cost and team skills.

**Q6. What are pre-filtering and post-filtering?**
Pre-filtering applies metadata filters before vector search: accurate, but it may be slow or bypass the ANN index. Post-filtering searches first and then filters: fast, but it may return fewer than k results. Modern engines integrate filtering into the ANN traversal.

**Q7. How do you measure an ANN index's quality?**
Recall@k against exact (flat) search on a sample of queries, plus latency percentiles (p50/p95/p99), throughput (QPS), memory use and build time.

---

## 15. Hands-on exercises

1. **Recall vs speed.** With FAISS, index 200k random normalised vectors in Flat, HNSW and IVF-PQ. Measure recall@10 vs Flat and query latency while varying `efSearch` and `nprobe`.
2. **Chroma CRUD.** Build a product catalogue with metadata filters (category, price range).
3. **pgvector + SQL.** Create `users` and `documents` tables and write one SQL query that returns the top 5 similar documents the user is allowed to see (join on permissions).
4. **Hybrid search.** Use Weaviate's `hybrid()` with `alpha` values 0, 0.5 and 1 on queries containing product codes. Compare the results.
5. **Swappable store.** Implement `PgVectorStore` with the `VectorStore` protocol and run the same RAG app against Chroma and pgvector.

### Assignment

Write a **vector database benchmark report**: load the same 50k–100k embedded documents (for example a Wikipedia subset) into Chroma, FAISS (HNSW) and pgvector. Measure ingestion time, query latency (p50/p95), recall@10 against exact search, and memory/disk usage. Present the results in tables and charts and recommend a store for three scenarios (prototype, SaaS, enterprise).

---

## 16. References

- [Chroma documentation](https://docs.trychroma.com/)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) · [FAISS documentation](https://faiss.ai/)
- [Pinecone documentation](https://docs.pinecone.io/)
- [Weaviate documentation](https://docs.weaviate.io/) · [Python client](https://docs.weaviate.io/weaviate/client-libraries/python)
- [pgvector on GitHub](https://github.com/pgvector/pgvector) · [pgvector-python](https://github.com/pgvector/pgvector-python)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Malkov & Yashunin — HNSW paper](https://arxiv.org/abs/1603.09320)
- [Jégou et al. — Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202)
- [ANN-Benchmarks](https://ann-benchmarks.com/)

---

| ← Previous | Index | Next → |
|---|---|---|
| [07. Retrieval-Augmented Generation](07-retrieval-augmented-generation.md) | [Home](README.md) | [09. Fine-Tuning LLMs](09-fine-tuning-llms.md) |
