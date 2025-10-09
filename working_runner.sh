#!/bin/bash

# Fix permissions first
echo "🔧 Setting up environment..."
mkdir -p logs
chmod 755 logs
chmod 644 *.py *.pkl

# Kill any existing bot processes
echo "🛑 Stopping any existing bots..."
pkill -f "python.*bot" 2>/dev/null

# Start bots with proper error handling
echo "🚀 Starting trading bots..."

start_bot() {
    local name=$1
    local script=$2
    local logfile="logs/${name}.log"
    
    if [ -f "$script" ]; then
        echo "Starting $name bot..."
        python "$script" > "$logfile" 2>&1 &
        local pid=$!
        echo $pid > "logs/${name}.pid"
        echo "✅ $name bot started (PID: $pid)"
        return 0
    else
        echo "❌ Script not found: $script"
        return 1
    fi
}

# Start each bot with delay
start_bot "ml_forced" "ml_forced_signals_bot.py"
sleep 3
start_bot "technical" "technical_scalping_bot.py" 
sleep 3
start_bot "realtime" "realtime_monitoring_bot.py"
sleep 3
start_bot "highfreq" "high_frequency_bot.py"

echo ""
echo "📊 Bot startup complete"
echo "📁 Logs: ls -la logs/"
echo "🔍 Check running: ps aux | grep python | grep bot"
echo "🛑 Stop all: pkill -f python"
