import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="komma_ai",
    version="0.1.1",
    author="Philipp Huismann",
    author_email="phil.huismann@gmail.com",
    description="A package to automatically put commas where they belong with machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FruitVul/komma_ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)