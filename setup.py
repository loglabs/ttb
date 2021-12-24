from setuptools import setup, find_packages

import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="ttb",
    version="0.1",
    description="Tabular Time Series Benchmark",
    long_description=README,
    long_description_content_type="text/markdown",
    author="shreyashankar",
    author_email="shreya@cs.stanford.edu",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "psycopg2-binary",
        "pandas",
        "python-dotenv",
    ],
    include_package_data=True,
)
