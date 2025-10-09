#!/usr/bin/env python3
"""
Fixed Bot Runner - Handles permission issues
"""

import subprocess
import sys
import time
import os
import signal

def setup_environment():
    """Setup directories and fix permissions"""
    try:
        # Create logs directory
        if not os.path.exists("logs"):
            os.makedirs("logs", mode=0o755)
            print("✅ Created logs directory")
        
        # Ensure write permissions
        os.chmod("logs", 0o755)
        return True
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def start_bots():
    print("🚀 Starting All 4 Trading Bots...")
    print("=" * 50)
    
    if not setup_environment():
        print("❌ Cannot setup environment. Check permissions.")
        return []
    
    bots = [
        ("ML Forced", "ml_forced_signals_bot.py", "ml_forced.log"),
        ("Technical", "technical_scalping_bot.py", "technical.log"), 
        ("Realtime", "realtime_monitoring_bot.py", "realtime.log"),
        ("HighFreq", "high_frequency_bot.py", "highfreq.log")
    ]
    
    processes = []
    
    for name, script, log_file in bots:
        try:
            if not os.path.exists(script):
                print(f"❌ {name}: Script '{script}' not found")
                continue
                
            log_path = f"logs/{log_file}"
            print(f"Starting {name} bot...")
            
            # Start the bot process
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            
            processes.append((name, process, log_path))
            print(f"✅ {name} bot started (PID: {process.pid})")
            time.sleep(3)  # Stagger startup
            
        except PermissionError as e:
            print(f"❌ {name}: Permission denied - {e}")
        except Exception as e:
            print(f"❌ {name}: Failed to start - {e}")
    
    print(f"\n📊 Successfully started {len(processes)}/4 bots")
    
    if processes:
        print("\n📁 Log Files:")
        for name, process, log_path in processes:
            print(f"  {name}: {log_path}")
        
        print("\n🔍 Monitor logs:")
        print("  tail -f logs/ml_forced.log")
        print("  tail -f logs/technical.log")
        print("  tail -f logs/*.log")
        
        print("\n🛑 Stop all bots:")
        print("  pkill -f python")
    
    return processes

def stop_bots():
    print("🛑 Stopping all trading bots...")
    os.system("pkill -f 'python.*bot' 2>/dev/null")
    time.sleep(2)
    print("✅ All bots stopped")

def check_status():
    print("🤖 Bot Status Check:")
    print("=" * 30)
    
    # Check running processes
    result = subprocess.run(
        "ps aux | grep -E 'python.*bot' | grep -v grep", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        print(f"🟢 Running: {len(lines)} bots")
        for line in lines:
            parts = line.split()
            pid = parts[1]
            cmd = ' '.join(parts[10:])
            # Extract bot name from command
            if 'ml_forced' in cmd:
                name = "ML Forced"
            elif 'technical' in cmd:
                name = "Technical" 
            elif 'realtime' in cmd:
                name = "Realtime"
            elif 'high_frequency' in cmd:
                name = "HighFreq"
            else:
                name = "Unknown"
            print(f"  {name} (PID: {pid})")
    else:
        print("🔴 No bots running")
    
    # Check log files
    print("\n📁 Log Files:")
    if os.path.exists("logs"):
        logs = os.listdir("logs")
        if logs:
            for log in sorted(logs):
                if log.endswith('.log'):
                    size = os.path.getsize(f"logs/{log}")
                    print(f"  {log} ({size} bytes)")
        else:
            print("  No log files yet")
    else:
        print("  logs/ directory not found")

def run_in_background():
    """Run bots in background and exit script"""
    print("🚀 Starting bots in background mode...")
    
    if not setup_environment():
        return False
    
    bots = [
        ("ml_forced_signals_bot.py", "logs/ml_forced.log"),
        ("technical_scalping_bot.py", "logs/technical.log"), 
        ("realtime_monitoring_bot.py", "logs/realtime.log"),
        ("high_frequency_bot.py", "logs/highfreq.log")
    ]
    
    started = 0
    for script, log_file in bots:
        try:
            if os.path.exists(script):
                # Start in background using nohup
                cmd = f"nohup python {script} > {log_file} 2>&1 &"
                os.system(cmd)
                started += 1
                print(f"✅ Started {script}")
                time.sleep(2)
            else:
                print(f"❌ Script not found: {script}")
        except Exception as e:
            print(f"❌ Failed to start {script}: {e}")
    
    print(f"\n📊 Started {started}/4 bots in background")
    print("💡 Run 'python fixed_bot_runner.py status' to check status")
    return started > 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "start"
    
    if action == "start":
        # Run in foreground (keeps monitoring)
        processes = start_bots()
        if processes:
            try:
                print("\n⏳ Monitoring bots... Press Ctrl+C to stop all.")
                while True:
                    time.sleep(10)
                    # Check status
                    alive = [(name, p, log) for name, p, log in processes if p.poll() is None]
                    if len(alive) < len(processes):
                        print(f"⚠️  {len(processes) - len(alive)} bot(s) crashed")
                        processes = alive
                    if not processes:
                        print("💥 All bots have stopped")
                        break
            except KeyboardInterrupt:
                print("\n🛑 User interrupted - stopping bots...")
                stop_bots()
                
    elif action == "background":
        # Run in background and exit
        run_in_background()
        
    elif action == "stop":
        stop_bots()
        
    elif action == "status":
        check_status()
        
    elif action == "logs":
        # Show recent logs
        if os.path.exists("logs"):
            for log_file in ["ml_forced.log", "technical.log", "realtime.log", "highfreq.log"]:
                path = f"logs/{log_file}"
                if os.path.exists(path):
                    print(f"\n📄 {log_file}:")
                    print("=" * 40)
                    os.system(f"tail -5 {path}")
                else:
                    print(f"📄 {log_file}: Not found")
        else:
            print("❌ logs directory not found")
        
    else:
        print("Usage: python fixed_bot_runner.py [start|background|stop|status|logs]")
        print("  start     - Start bots in foreground with monitoring")
        print("  background- Start bots in background and exit")
        print("  stop      - Stop all running bots")
        print("  status    - Check bot status and logs")
        print("  logs      - Show recent log output")
