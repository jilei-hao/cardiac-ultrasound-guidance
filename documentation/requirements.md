# Dependencies and Version Management

This document provides information about managing dependencies for the lumify_streaming project.

## Quick Start

Install all dependencies at once using the requirements file:

```bash
pip install -r requirements.txt
```

## Checking Package Versions

### Check NumPy Version

There are several ways to check the installed version of NumPy:

#### Method 1: Using pip
```bash
pip show numpy
```

This will display detailed information including the version number.

#### Method 2: Using Python command line
```bash
python -c "import numpy; print(numpy.__version__)"
```

#### Method 3: Within a Python script
```python
import numpy as np
print(f"NumPy version: {np.__version__}")
```

#### Method 4: Interactive Python shell
```bash
python
>>> import numpy
>>> numpy.__version__
'1.26.4'
```

### Check All Installed Package Versions

To see versions of all installed packages:

```bash
pip list
```

To see versions of packages used in this project:

```bash
pip freeze | grep -E "numpy|opencv-python|torch|Pillow|matplotlib|mss|pyobjc|requests"
```

## Dependency List

The project requires the following packages:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ==1.26.4 | Numerical computing and array operations |
| opencv-python | ==4.9.0.80 | Computer vision and image processing |
| torch | ==2.1.2 | Deep learning framework |
| torchvision | ==0.16.2 | Computer vision models and utilities |
| Pillow | ==10.2.0 | Image manipulation |
| matplotlib | ==3.8.3 | Data visualization |
| mss | ==9.0.1 | Screen capture utility |
| pyobjc-* | ==10.1 | macOS system integration |
| requests | ==2.31.0 | HTTP library |

## Version Constraints

### NumPy Version
The project requires NumPy version < 2.0 due to compatibility requirements with PyTorch 2.1.2. NumPy 2.0 introduced breaking changes that are not yet fully compatible with this version of PyTorch.

**Important:** Always ensure NumPy version is below 2.0:
```bash
pip install "numpy<2"
```

### Python Version
This project requires Python 3.12. Create the environment with:

```bash
conda create -n lumify-streaming python=3.12
conda activate lumify-streaming
```

## Updating Dependencies

### Update a Single Package

To update a specific package to the latest compatible version:

```bash
pip install --upgrade numpy
```

**Note:** Be careful when updating packages individually, as this may introduce incompatibilities. Always test after updates.

### Regenerate requirements.txt

If you've made changes to the dependencies, regenerate the requirements file:

```bash
pip freeze > requirements.txt
```

Then manually edit to keep only the necessary packages.

## Troubleshooting

### NumPy Version Conflicts

If you encounter errors related to NumPy version:

1. Check current version:
   ```bash
   pip show numpy
   ```

2. Uninstall and reinstall with correct version:
   ```bash
   pip uninstall numpy
   pip install "numpy<2"
   ```

3. Verify installation:
   ```bash
   python -c "import numpy; print(numpy.__version__)"
   ```

### Missing Dependencies

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Platform-Specific Issues

#### macOS
The `pyobjc` packages are required for screen capture on macOS. If you encounter installation issues:

```bash
pip install --upgrade pip
pip install pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-Quartz
```

#### Other Platforms
Screen capture functionality may require different dependencies on Windows or Linux. Refer to the `mss` documentation for platform-specific requirements.

## Development Setup

For a complete development environment setup:

1. Create conda environment:
   ```bash
   conda create -n lumify-streaming python=3.12
   conda activate lumify-streaming
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "import numpy, cv2, torch, PIL; print('All core dependencies installed successfully')"
   ```

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [pip Documentation](https://pip.pypa.io/en/stable/)
