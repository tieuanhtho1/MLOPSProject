# 1. Start from a base Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install packages
# This is done first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your entire project (app.py, templates, models, etc.)
COPY . .

# 5. Expose the port Gunicorn will run on
EXPOSE 8000

# 6. Define the command to run your app
# This tells Gunicorn to find the 'app' object inside the 'app.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]