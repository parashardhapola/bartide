from setuptools import setup, find_packages

setup(
    name="bartide",
    version=open("VERSION").readline().rstrip("\n"),
    description="A Python package to extract, correct and analyze nucleotide barcodes from sequenced reads.",
    packages=find_packages(),
    author="Parashar Dhapola",
    author_email="parashar.dhapola@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/parashardhapola/bartide",
    include_package_data=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Data analysis",
        "Topic :: Genomics",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "nmslib",
        "matplotlib",
        "loguru",
        "editdistance",
        "tqdm",
        "pandas",
        "joblib",
    ],
    extras_require={"dev": ["pytest", "pytest-pep8", "pytest-cov", "black"]},
    keywords=["Text Mining", "Barcode", "nucleotide", "sequencing", "deduplicate"],
)
