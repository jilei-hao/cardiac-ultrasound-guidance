"""
Setup script for cardiac-ultrasound-guidance package.

Install with:
    pip install .                  # Normal install
    pip install -e .               # Editable/development install
    pip install .[dev]             # With development dependencies
"""

from setuptools import setup, find_packages

# Core dependencies
install_requires = [
    # Core numerical and imaging
    "numpy",
    "opencv-python",
    "mss",
    "Pillow",
    "matplotlib",
    "pandas",
    
    # Deep learning
    "torch",
    "torchvision",
    
    # Rendering
    "pyrender",
    "trimesh",
    "cairosvg",
    
    # GUI / Cross-platform windowing
    "PySide6",
    
    # Utilities
    "requests",
    "python-dotenv>=1.0.0",
    
    # macOS window management (required for screen capture helpers)
    "pyobjc==10.3 ; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz ; sys_platform == 'darwin'",
]

# Development/test dependencies
dev_requires = [
    "pytest",
    "pytest-cov",
    "black",
    "flake8",
    "mypy",
]

setup(
    name="cardiac-ultrasound-guidance",
    version="0.1.0",
    description="Real-time cardiac ultrasound guidance system with deep learning inference",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    author="Paradigm Team",
    author_email="",
    url="https://github.com/paradigm/cardiac-ultrasound-guidance",
    license="MIT",
    
    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.stl", "*.png", "*.jpg"],
    },
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
    
    # Console script entry points
    entry_points={
        "console_scripts": [
            # Main guidance application
            "cusg-main=main:main",
            # Guidance UI application
            "cusg-guidance-ui=guidance_ui:main",
            # IMU recording utility
            "cusg-record-imu=record_imu:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    
    # Keywords for discoverability
    keywords="ultrasound, cardiac, guidance, deep-learning, medical-imaging",
)
