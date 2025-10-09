from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sdg-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MLOps pipeline for SDG classification of research papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/SDGClassifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
        ],
        "api": [
            "flask>=2.2.0",
            "gunicorn>=20.1.0",
            "requests>=2.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sdg-preprocess=src.data.preprocess:main",
            "sdg-train=src.models.train:main",
            "sdg-evaluate=src.evaluation.evaluate:main",
            "sdg-export=src.models.export:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    include_package_data=True,
    keywords="machine-learning, nlp, sdg, classification, mlops, dvc",
    project_urls={
        "Bug Reports": "https://github.com/username/SDGClassifier/issues",
        "Source": "https://github.com/username/SDGClassifier",
        "Documentation": "https://github.com/username/SDGClassifier#readme",
    },
)