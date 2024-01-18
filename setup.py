import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Pyspady",
    version="0.0.1",
    author="Michael Paglia",
    author_email="michaelpagliadev@gmail.com",
    description="A Python sparse multi dictionary coding library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petkobogdanov/pyspady",
    packages=setuptools.find_packages(),
    install_requires=[
        'scipy'
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
