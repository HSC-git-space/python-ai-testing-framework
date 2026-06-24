# Base image — Python 3.11 slim keeps the image size small
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If requirements.txt hasn't changed, pip install is skipped on rebuild
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Create reports directory inside container
RUN mkdir -p reports

# Default command — runs the full test suite
CMD ["python", "-m", "pytest", "tests/", "--html=reports/report.html", "--self-contained-html", "-v"]