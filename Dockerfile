FROM python:3.11.13

# Build argument to control Claude Code installation
ARG INSTALL_CLAUDE=true

RUN apt update
RUN apt upgrade -y
# Install system dependencies for OpenCV
RUN apt install -y libgl1 libglib2.0-0
RUN pip install --upgrade pip
RUN pip install numpy numba matplotlib pandas scipy scikit-image
RUN pip install opencv-python pydicom flet pyvista

# Install Node.js and Claude Code (if enabled)
RUN if [ "$INSTALL_CLAUDE" = "true" ]; then \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs && \
    npm install -g @anthropic-ai/claude-code; \
    fi

WORKDIR /home/
CMD ["/bin/bash"]