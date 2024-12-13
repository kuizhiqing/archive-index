import os
import uuid

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from qdrant_client import QdrantClient
from qdrant_client.http import models

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class config:
    chunk_size = 1000
    chunk_overlap = 100
    host = "<host>"
    port = 6333
    vec_size = 1024
    collection_name = "test"
    model_name = "dunzhang/stella_en_1.5B_v5"

    root_dir = "<dir>"
    skip_dirs = [".git"]
    extensions = (".pdf",)

    log_files = "indexed_files.txt"


def main():
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap
    host = config.host
    port = config.port
    vec_size = config.vec_size
    collection_name = config.collection_name
    model_name = config.model_name
    root_dir = config.root_dir
    skip_dirs = config.skip_dirs
    extensions = config.extensions
    log_files = config.log_files

    assert isinstance(extensions, tuple)

    log_file = open(log_files, "r+")
    indexed_files = log_file.read().splitlines()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    model = SentenceTransformer(model_name)

    client = QdrantClient(host=host, port=port)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vec_size, distance=models.Distance.COSINE),
        )

    logger.info(f"process dir {root_dir}")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        files = [f for f in filenames if f.endswith(extensions)]
        for filename in files:
            relpath = os.path.relpath(dirpath, root_dir)
            relname = os.path.join(relpath, filename)
            file_path = os.path.join(root_dir, relpath, filename)
            if relname in indexed_files:
                logger.info(f"skip file {file_path}")
                continue
            else:
                logger.info(f"processing {file_path}")

            loader = PyPDFLoader(file_path)
            try:
                document = loader.load()
            except Exception as e:
                logger.error(f"load file {filename} error: {e}")
                continue

            documents = text_splitter.split_documents(document)
            logger.info(f"split {filename} into {len(documents)} chunks")
            points = []
            for i, doc in enumerate(documents):
                embedding = model.encode(doc.page_content, convert_to_tensor=True)
                payload = {
                    "filename": filename,
                    "relpath": relpath,
                    "index": i,
                    "type": "pdf",
                    "page": doc.metadata.get("page", -1),
                }
                uid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}-{i}")
                point = models.PointStruct(id=uid.hex, vector=embedding.tolist(), payload=payload)
                logger.info(f"add point with {payload}")
                points.append(point)

            try:
                if len(points) > 0:
                    client.upsert(collection_name=collection_name, points=points)
            except Exception as e:
                logger.error(f"add points error: {e}")
            log_file.write(f"{relname}\n")
            log_file.flush()
            logger.info(f"finished processing {file_path}")

    log_file.close()


if __name__ == "__main__":
    main()
