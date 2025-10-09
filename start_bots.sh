#!/bin/bash

# Create logs directory
mkdir -p logs

echo "🚀 Starting Crypto Trading Bots"
echo "=================================================="

# Start all bots in background
python ml_forced_signals_bot.py > logs/ml_bot.log 2>&1 &
ML_PID=$!
echo "✅ ML Forced Signals Bot started (PID: $ML_PID)"

python technical_scalping_bot.py > logs/technical_bot.log 2>&1 &
TECH_PID=$!
echo "✅ Technical Scalping Bot started (PID: $TECH_PID)"

python realtime_monitoring_bot.py > logs/realtime_bot.log 2>&1 &
REALTIME_PID=$!
echo "✅ Real-time Monitoring Bot started (PID: $REALTIME_PID)"

python high_frequency_bot.py > logs/highfreq_bot.log 2>&1 &
HIGHFREQ_PID=$!
echo "✅ High Frequency Bot started (PID: $HIGHFREQ_PID)"

echo ""
echo "📊 All 4 bots running in background!"
echo "📁 Check logs: tail -f logs/*.log"
echo "🛑 To stop all: pkill -f python"
echo ""
echo "Bot PIDs:"
echo "ML Forced: $ML_PID"
echo "Technical: $TECH_PID" 
echo "Realtime: $REALTIME_PID"
echo "HighFreq: $HIGHFREQ_PID"
