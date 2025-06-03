from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        # For example:
        # "requests>=2.25.1",
        # "pandas>=1.2.0",
    ],
    author="Wei Yang",
    author_email="weiyang2048@gmail.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/weiyang2048/stonk_trading",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
