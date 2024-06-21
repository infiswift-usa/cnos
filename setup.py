from setuptools import setup, find_packages

setup(
    name='cnos',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'omegaconf',
        'torchmetrics',
        'fvcore',
        'iopath',
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
        'distinctipy',
        'scikit-image',
        'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@main'
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
    package_data={'cnos':['configs/**/*'], 'cnos.poses':['predefined_poses/*']},
    zip_safe=False,
    include_package_data=True
)

