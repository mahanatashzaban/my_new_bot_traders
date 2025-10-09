#!/usr/bin/env python3
"""
Orchestrator to Run All 4 Trading Bots Simultaneously
"""

import subprocess
import sys
import time
import os
import signal
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BotOrchestrator')

class BotManager:
    def __init__(self):
        self.bots = {
            'ml_forced': {
                'script': 'ml_forced_signals_bot.py',
                'process': None,
                'log_file': 'logs/ml_forced_bot.log'
            },
            'technical': {
                'script': 'technical_scalping_bot.py', 
                'process': None,
                'log_file': 'logs/technical_bot.log'
            },
            'realtime': {
                'script': 'realtime_monitoring_bot.py',
                'process': None,
                'log_file': 'logs/realtime_bot.log'
            },
            'highfreq': {
                'script': 'high_frequency_bot.py',
                'process': None,
                'log_file': 'logs/highfreq_bot.log'
            }
        }
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs('logs', exist_ok=True)
        os.makedirs('pids', exist_ok=True)
        
    def start_bot(self, bot_name):
        """Start a single bot"""
        bot_info = self.bots[bot_name]
        
        try:
            # Create log file
            log_file = open(bot_info['log_file'], 'w')
            
            # Start bot process
            process = subprocess.Popen(
                [sys.executable, bot_info['script']],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group
            )
            
            bot_info['process'] = process
            bot_info['start_time'] = datetime.now()
            
            # Save PID to file
            with open(f'pids/{bot_name}_bot.pid', 'w') as f:
                f.write(str(process.pid))
            
            logger.info(f"✅ Started {bot_name} bot (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {bot_name} bot: {e}")
            return False
    
    def start_all_bots(self):
        """Start all four bots"""
        logger.info("🚀 STARTING ALL 4 TRADING BOTS")
        logger.info("=" * 50)
        
        success_count = 0
        for bot_name in self.bots:
            if self.start_bot(bot_name):
                success_count += 1
            time.sleep(2)  # Stagger startup
        
        logger.info(f"📊 Successfully started {success_count}/4 bots")
        return success_count
    
    def stop_bot(self, bot_name):
        """Stop a single bot"""
        bot_info = self.bots[bot_name]
        
        if bot_info['process'] and bot_info['process'].poll() is None:
            try:
                # Kill entire process group
                os.killpg(os.getpgid(bot_info['process'].pid), signal.SIGTERM)
                bot_info['process'].wait(timeout=10)
                logger.info(f"⏹️  Stopped {bot_name} bot")
            except:
                try:
                    os.killpg(os.getpgid(bot_info['process'].pid), signal.SIGKILL)
                    logger.info(f"🔫 Force killed {bot_name} bot")
                except:
                    logger.error(f"❌ Failed to stop {bot_name} bot")
        
        # Clean up PID file
        pid_file = f'pids/{bot_name}_bot.pid'
        if os.path.exists(pid_file):
            os.remove(pid_file)
    
    def stop_all_bots(self):
        """Stop all running bots"""
        logger.info("🛑 STOPPING ALL TRADING BOTS")
        
        for bot_name in self.bots:
            self.stop_bot(bot_name)
        
        logger.info("✅ All bots stopped")
    
    def check_bot_status(self):
        """Check status of all bots"""
        status = {}
        for bot_name, bot_info in self.bots.items():
            if bot_info['process']:
                if bot_info['process'].poll() is None:
                    status[bot_name] = 'RUNNING'
                else:
                    status[bot_name] = 'STOPPED'
            else:
                status[bot_name] = 'NOT STARTED'
        return status
    
    def monitor_bots(self, duration_hours=4):
        """Monitor bots and restart if they crash"""
        logger.info(f"🔍 Monitoring bots for {duration_hours} hours...")
        
        end_time = time.time() + (duration_hours * 3600)
        check_interval = 60  # Check every minute
        
        while time.time() < end_time:
            try:
                status = self.check_bot_status()
                running_bots = [name for name, stat in status.items() if stat == 'RUNNING']
                
                logger.info(f"🤖 Active bots: {len(running_bots)}/4 - {', '.join(running_bots)}")
                
                # Restart any crashed bots
                for bot_name, stat in status.items():
                    if stat == 'STOPPED':
                        logger.warning(f"🔄 Restarting crashed bot: {bot_name}")
                        self.start_bot(bot_name)
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(check_interval)
        
        self.stop_all_bots()

def main():
    """Main function"""
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = 'start'
    
    manager = BotManager()
    
    if action == 'start':
        print("🚀 Starting All Trading Bots...")
        success_count = manager.start_all_bots()
        
        if success_count > 0:
            print(f"\n✅ Successfully started {success_count} bots")
            print("📁 Logs: tail -f logs/*.bot.log")
            print("📊 Status: python run_all_bots.py status")
            print("🛑 Stop: python run_all_bots.py stop")
            print("\n💡 Run in background: nohup python run_all_bots.py start &")
            
            # Start monitoring
            try:
                manager.monitor_bots(duration_hours=4)
            except KeyboardInterrupt:
                manager.stop_all_bots()
        else:
            print("❌ Failed to start any bots")
    
    elif action == 'stop':
        manager.stop_all_bots()
    
    elif action == 'status':
        status = manager.check_bot_status()
        print("\n🤖 BOT STATUS:")
        print("=" * 30)
        for bot_name, stat in status.items():
            status_icon = "🟢" if stat == 'RUNNING' else "🔴" if stat == 'STOPPED' else "⚪"
            print(f"{status_icon} {bot_name:12} : {stat}")
    
    elif action == 'restart':
        print("🔄 Restarting all bots...")
        manager.stop_all_bots()
        time.sleep(5)
        manager.start_all_bots()
    
    else:
        print("Usage: python run_all_bots.py [start|stop|status|restart]")

if __name__ == "__main__":
    main()
