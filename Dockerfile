FROM python:3.9-slim

# Create a non-root user as required by Hugging Face
RUN useradd -m -u 1000 user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory and user
WORKDIR /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements file (with correct permissions)
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the code (with correct permissions)
COPY --chown=user . /app

# Configure Streamlit theme
ENV STREAMLIT_THEME="light"
ENV STREAMLIT_PRIMARY_COLOR="#4169E1"
ENV STREAMLIT_BACKGROUND_COLOR="#FFFFFF"
ENV STREAMLIT_TEXT_COLOR="#262730"

# Expose the port Hugging Face expects
EXPOSE 7860

# Command to run the application (note the port change to 7860)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]