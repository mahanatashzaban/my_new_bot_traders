FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./
COPY *.pkl ./

# Create directory for logs
RUN mkdir -p /app/logs

# Run all bots simultaneously
CMD ["sh", "-c", "python ml_forced_signals_bot.py & python technical_scalping_bot.py & python realtime_monitoring_bot.py & python high_frequency_bot.py & wait"]
