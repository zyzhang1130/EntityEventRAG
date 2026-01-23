import setuptools
from pathlib import Path


def read_long_description():
    """Read the long description from README.md."""
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "Entity-Event RAG: Bipartite Graph-based Retrieval-Augmented Generation"


def retrieve_metadata():
    """Retrieve metadata from __init__.py."""
    vars2find = ["__author__", "__version__", "__url__"]
    vars2readme = {}
    try:
        with open("./entity_event_rag/__init__.py") as f:
            for line in f.readlines():
                for v in vars2find:
                    if line.startswith(v):
                        line = (
                            line.replace(" ", "")
                            .replace('"', "")
                            .replace("'", "")
                            .strip()
                        )
                        vars2readme[v] = line.split("=")[1]
    except FileNotFoundError:
        raise FileNotFoundError("Metadata file './entity_event_rag/__init__.py' not found.")

    missing_vars = [v for v in vars2find if v not in vars2readme]
    if missing_vars:
        raise ValueError(
            f"Missing required metadata variables in __init__.py: {missing_vars}"
        )

    return vars2readme


def read_requirements():
    """Read dependencies from requirements.txt."""
    deps = []
    try:
        with open("./requirements.txt") as f:
            deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(
            "Warning: 'requirements.txt' not found. No dependencies will be installed."
        )
    return deps


metadata = retrieve_metadata()
long_description = read_long_description()
requirements = read_requirements()

setuptools.setup(
    name="entity-event-rag",
    url=metadata["__url__"],
    version=metadata["__version__"],
    author=metadata["__author__"],
    description="Entity-Event RAG: Bipartite Graph-based Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=("tests*", "docs*", "examples*", "experiments*", "scripts*", "reproduce*")
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "RAG",
        "retrieval-augmented-generation",
        "knowledge-graph",
        "entity-extraction",
        "event-extraction",
        "NLP",
        "LLM",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
        ],
        "visualization": [
            "pyvis>=0.3",
            "plotly>=5.0",
        ],
    },
    include_package_data=True,
    project_urls={
        "Documentation": metadata.get("__url__", ""),
        "Source": metadata.get("__url__", ""),
        "Tracker": f"{metadata.get('__url__', '')}/issues"
        if metadata.get("__url__")
        else "",
    },
    entry_points={
        "console_scripts": [
            "entity-event-rag=entity_event_rag.cli:main",
        ],
    },
)
