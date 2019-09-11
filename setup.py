import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bash-org-sentiment-analyser",
    version="0.0.5",
    author="Andrew Sorokin",
    author_email="i40mines@yandex.ru",
    description="Attempt to classify quotes from bash.im",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/40min/bash-org-sentiment-analyser.git",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3',
    install_requires=[
        'scikit-learn',
        'nltk',
        'pandas~=0.24.1',
        'pandas~=0.24.1',
    ],
)