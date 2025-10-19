sudo docker build -t "nlstat-service" -f Dockerfile.service .

sudo docker run -p 8000:8000 -e "GEMINI_API_KEY=$GEMINI_API_KEY" nlstat-service