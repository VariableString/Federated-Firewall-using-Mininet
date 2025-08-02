from setuptools import setup, find_packages

setup(
    name="federated-firewall-mininet",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.11.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scapy>=2.4.5",
        "mininet>=2.3.0",
        "PyYAML>=6.0",
        "psutil>=5.8.0",
        "colorlog>=6.6.0",
        "ryu>=4.34",
        "eventlet>=0.33.0"
    ],
    author="VariableString",
    description="Federated Learning Distributed Firewall for Zero Trust Networks with Mininet",
    python_requires=">=3.8"
)