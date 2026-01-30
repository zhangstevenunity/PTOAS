from setuptools import setup, find_namespace_packages

setup(
    name="ptoas",
    version="0.1.0",
    description="PTO Assembler & Optimizer",
    # NOTE: find_namespace_packages detects folders even without __init__.py
    packages=find_namespace_packages(),
    # NOTE: The * at the end captures .so.22, .so.22.1, etc.
    package_data={
            "mlir": [
                "**/*.so*",
                "**/*.pyd",
                "**/*.py",
                "_mlir_libs/*.so*",
                "bin/ptoas",
            ],
        },
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    # Add ptoas as a console script entry point using the wrapper
    entry_points={
        "console_scripts": [
            "ptoas=mlir.bin.ptoas_wrapper:main",
        ],
    },
)
