from setuptools import setup, find_packages

setup(
    name="alpha_pose",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # FastAPI 관련
        "fastapi",
        "uvicorn",
        "websockets",
        "python-multipart",
        "pydantic",
        "pydantic-settings",

        # 이미지 처리 관련
        "opencv-python",
        "numpy",
        "Pillow",

        # PyTorch & YOLO 관련
        "torch",
        "torchvision",
        "ultralytics",

        # LangChain 관련
        "langchain",
        "langchain-core",
        "langgraph",
        "langchain-openai",
        "langchain-community",

        # Neo4j 관련
        "neo4j",

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
    ],
    python_requires=">=3.9",
    author="leeloun",
    author_email="leeloun@naver.com",
    description="Action recognition system using pose estimation and LangChain",
    keywords="pose-estimation, action-recognition, langchain, yolo",
) 