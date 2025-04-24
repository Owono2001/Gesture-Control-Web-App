# Dockerfile for Gesture Control Web App (with xauth fix)

# 1. Base Image
FROM python:3.10-slim

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# 3. Create & Set Working Directory
WORKDIR $APP_HOME

# 4. Install System Dependencies
#    - Added 'xauth' which is needed by xvfb-run
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Playsound dependencies
    alsa-utils \
    libasound2-plugins \
    # PyAutoGUI/xvfb dependencies
    xvfb \
    xauth \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Expose Port
EXPOSE 5000

# 8. Define Command to Run
#    Using xvfb-run for pyautogui, gunicorn for Flask/SocketIO
#    Remember the limitations for camera/pyautogui on platforms like Render.
CMD ["xvfb-run", "gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "--log-level", "debug", "app:app"]
