from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Mentevo",
    version="0.0.0",
    description="Personal toolbox to simulate Cognitive Flexibility and Stability",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Alessandra Brondetta",
    author_email="alessandra.brondetta@gmail.com",
    license="MIT",
    install_requires=['numpy', 'matplotlib', 'scipy'],
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
