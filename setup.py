import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Pyspady",
    version="0.0.1",
    author="Michael Paglia, Proshanto Dabnath, Michael Smith, Joseph Regan",
    author_email="michaelpagliadev@gmail.com",
    description="A Python sparse multi dictionary coding library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petkobogdanov/pyspady",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',            # Linear algebra operations
        'scipy',            # Linear algebra operations
        'tensorly',         # Tensor operations
        'matplotlib',       # Plotting
        'pandas',           # Data storage & retrieval
        'pyfiglet',         # Logo
        'geopandas',        # For geospatial data operations
        'contextily',       # For basemap tile sourcing
        'shapely',          # For geometric operations
        'geopy',            # For geocoding and other geographic computations
        'scikit-learn',     # For ParameterSampler, BaseEstimator, TransformerMixin
        'pyproj',           # Coordinate transformations used in demo modules
        'scikit-learn'      # "Smart-search" modules
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
