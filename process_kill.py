import psutil
from datetime import datetime, timedelta
import signal
import time



# 5시간 이상 실행 중이고 CPU 사용률이 50% 이상인 프로세스 찾기
while True:
    # 현재 시간
    now = datetime.now()
    # 2시간 전 시간
    threshold_time = now - timedelta(hours=2)
    for proc in psutil.process_iter(['pid', 'create_time', 'cpu_percent', 'status', 'cmdline']):
        try:
            # 프로세스 정보 가져오기
            pid = proc.info['pid']
            create_time = datetime.fromtimestamp(proc.info['create_time'])
            initial_cpu_percent = proc.info['cpu_percent']
            status = proc.info['status']
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else "N/A"
         
            # 프로세스가 실행 중인지 확인
            if status == psutil.STATUS_RUNNING and 'booksim2' in cmdline:
                # 5시간 이상 실행 중인지 확인
                if create_time <= threshold_time:
                    # CPU 사용률 업데이트를 위해 잠시 대기
                    time.sleep(0.1)
                    proc.cpu_percent(interval=0.1)
                    updated_cpu_percent = proc.cpu_percent(interval=0.1)

                    proc.kill()
                    print(f"PID: {pid}, CPU: {updated_cpu_percent}%, Start Time: {create_time}, Command: {cmdline} killed")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    time.sleep(60)
