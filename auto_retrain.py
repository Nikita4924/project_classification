import schedule
import time
import subprocess
import os

# === Базовая директория проекта ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Путь к скрипту retrain.py ===
RETRAIN_SCRIPT = os.path.join(BASE_DIR, "bot", "retrain.py")

# === Задача: запуск retrain.py ===
def run_retrain():
    print("🧠 Запускаем переобучение модели...")
    try:
        subprocess.run(["python", RETRAIN_SCRIPT], check=True)
        print("✅ Переобучение завершено успешно.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при запуске retrain.py: {e}")

# === Расписание: каждый день в 09:00 ===
schedule.every().day.at("09:00").do(run_retrain)

print("⏱️ Автозапуск retrain.py активирован. Ожидаем расписание...")

# === Основной цикл планировщика ===
while True:
    schedule.run_pending()
    time.sleep(30)