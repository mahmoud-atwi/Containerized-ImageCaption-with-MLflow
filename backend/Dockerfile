# Set base image (Python 3.10)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy list of backend requirements
COPY requirements-backend.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy files to working directory (into a newly created /app directory)
COPY . /app

# Expose port
EXPOSE 8000

# Execute command when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]