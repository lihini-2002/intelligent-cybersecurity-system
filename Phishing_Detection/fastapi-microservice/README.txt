Local Setup Instructions

1. Clone the repo

git clone https://github.com/lihini-2002/intelligent-cybersecurity-system.git
cd Phishing_Detection/fastapi-microservice

2. Create and activate virtual environment

python -m venv venv        # Only once
source venv/bin/activate   # Run this every time

3. Install dependencies

pip install -r requirements.txt

4. Run the FastAPI service

uvicorn main:app --reload

5. Run with Docker (Production Setup)

docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
