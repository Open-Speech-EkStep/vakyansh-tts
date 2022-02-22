from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="vakyansh-tts",
    version="0.0.5",
    description="Text to speech for Indic languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Open-Speech-EkStep/vakyansh-tts.git",
    keywords="nlp, tts, Indic languages, deep learning, text to speech",
    # package_dir={'': 'src'},
    # packages=find_packages(where='src'),
    packages=["tts_infer"],
    python_requires=">=3.7, <4",
    install_requires=[
        "Cython==0.29.24",
        "layers==0.1.5",
        "librosa==0.8.1",
        "matplotlib==3.3.4",
        "numpy==1.20.2",
        "scipy==1.5.4",
        "tensorboardX==2.4",
        "tensorboard==2.7.0",
        "tqdm==4.62.3",
        "fastapi==0.70.0",
        "uvicorn==0.15.0",
        "gradio==2.5.2",
        "wavio==0.0.4",
        "pydload==1.0.9",
        "mosestokenizer==1.2.1",
        "indic-nlp-library==0.81"
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
)
