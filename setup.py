from setuptools import setup, find_packages

setup(
    name='keel',
    version="0.1.0",
    description="STL形式のファイルを出力可能な3Dモデリングツール",
    long_description="STL形式のファイルを出力可能な3Dモデリングツール\nコーディングで3Dモデルを作成する\nOpenGLベースのViewerも使用可能",
    url='',
    author='Sugai Takumi',
    author_email='',
    license='Free For Home Use',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Manufacturing",
        "License :: Free For Home Use",
        "Programming Language :: Python :: 3.7",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    keywords='3D Modeling',
    install_requires=["numpy", "PyOpenGL", "pygame"],
    packages=find_packages(exclude=('demo', 'docs')),
)
