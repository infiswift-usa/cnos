from setuptools import setup, find_packages

setup(
    name='cnos',
    version='0.1.0',
    packages=find_packages(),
    python_requires='==3.8',
    install_requires=[
        'torch',
        'torchvision',
        'omegaconf',
        'torchmetrics==0.10.3',
        'fvcore',
        'iopath',
        'xformers==0.0.18',
        'opencv-python',
        'pycocotools',
        'matplotlib',
        'onnxruntime',
        'onnx',
        'scipy',
        'ffmpeg',
        'hydra-colorlog',
        'hydra-core',
        'gdown',
        'pytorch-lightning',
        'pandas',
        'ruamel.yaml',
        'pyrender',
        'wandb',
        'distinctipy'
    ],
    dependency_links=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            # If you have any command line scripts, you can add them here.
            # 'script_name=module:function'
        ],
    },
)

