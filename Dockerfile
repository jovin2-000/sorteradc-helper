FROM sorteradc:v5

WORKDIR /helper

# sorteradc:v5 already has cv2, numpy, flask, matplotlib, PIL, etc.
RUN pip install --no-cache-dir --break-system-packages flask-cors 2>/dev/null || true

COPY backend/ /helper/backend/
COPY frontend/ /helper/frontend/

EXPOSE 8089

CMD ["python", "/helper/backend/app.py"]
