from setuptools import setup, find_packages

# 기본 라이브러리
install_requires = [
    # FastAPI 관련
    "fastapi",
    "uvicorn",
    "websockets",
    "python-multipart",
    "pydantic",
    "pydantic-settings",
    
    # --- 기타 라이브러리 ----
    # 이미지 처리 관련
    "opencv-python",
    "numpy",
    "Pillow",
    "ultralytics",

    # LangChain 관련
    "langchain",
    "langchain-core",
    "langgraph",
    "langchain-openai",
    "langchain-community",

    # Neo4j 관련
    "neo4j",
    "neo4j-driver",

    # 유틸리티
    "python-dotenv",
    "requests",
    "tqdm",
    "pandas",
    "matplotlib",
    
    # 개발 도구
    "pytest",
    "black",
    "isort",
    "mypy",
]

# PyTorch와 CUDA 관련 라이브러리
extras_require = {
    "pytorch-gpu": [
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        "torchaudio==2.1.2+cu118",
    ],
    "pytorch-cpu": [
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
    ],
    "mmaction": [
        "mmaction2==1.2.0",
        "mmcv==2.1.0",
        "mmengine==0.10.1",
    ],
}

setup(
    name="alpha_pose_backend",
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
    author="leeloun",
    author_email="leeloun@naver.com",
    description="Backend for AlphaPose project"
)
