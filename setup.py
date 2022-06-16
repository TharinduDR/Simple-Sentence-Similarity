from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simplests",
    version="2.1.1",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="Unsupervised models for Semantic Textual Similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharinduDR/Simple-Sentence-Similarity",
    packages=find_packages(exclude=("examples", )),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "numpy",
        "flair",
        "allennlp",
        "tqdm",
        "pyemd",
        "stop_words",
        "tensorflow_text",
        "tensorflow_hub",
        "sentence_transformers",
        "laserembeddings",
    ],
)
