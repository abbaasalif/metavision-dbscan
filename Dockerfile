FROM rapidsai/notebooks:23.08-cuda11.8-py3.9

# Switch to root user to install system packages
USER root

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Update system and install core dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    gnupg \
    software-properties-common \
    wget \
    curl \
    ca-certificates \
    libcanberra-gtk-module \
    mesa-utils \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Add GPG key for Ogre repository
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 81504C7DB60DF2F6
# Copy the APT source file for Metavision SDK
COPY metavision.list /etc/apt/sources.list.d/metavision.list

# Update package list to include Metavision SDK repository
RUN apt-get update

# Install Metavision SDK and Python bindings
RUN apt-get install -y --no-install-recommends \
    metavision-sdk \
    metavision-sdk-python3.9

# Ensure Metavision SDK is detected by Python
RUN conda env config vars set \
    PYTHONPATH=/usr/lib/python3/dist-packages:/opt/conda/lib/python3.9/site-packages \
    LD_LIBRARY_PATH=/usr/lib/python3/dist-packages:$LD_LIBRARY_PATH

# Install PyTorch with GPU support (CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional Python libraries
RUN pip install --no-cache-dir \
    opencv-python \
    pandas \
    matplotlib \
    scipy \
    scikit-learn 

# Set working directory
WORKDIR /workspace

# Ensure Metavision SDK is loaded properly
CMD ["bash", "-c", "export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/conda/lib/python3.9/site-packages && /bin/bash"]
