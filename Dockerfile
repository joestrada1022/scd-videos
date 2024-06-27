FROM tensorflow/tensorflow:2.15.0-gpu

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt