from setuptools import setup, find_packages

setup(
    name="ai-ml-generation-tasks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "pillow>=8.3.1",
    ],
)
