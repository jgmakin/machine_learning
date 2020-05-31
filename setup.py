import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="machine_learning",
    version="0.6.0",
    author="J.G. Makin",
    author_email="jgmakin@gmail.com",
    description="a collection of packages for ML projects, written in Tensorflow's Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgmakin/machine_learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'tensor2tensor==1.15.5',
        'tensorflow-probability>=0.7',
        'tfmpl',
        'protobuf>=3.7',
        # 'tensorflow-gpu==1.15.3'  the cpu version will also work
    ],
)
