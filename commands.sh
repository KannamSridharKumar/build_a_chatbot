
sudo kill -9 $(lsof -ti:8080)

docker build -t chainlit-app:latest .
docker run -p 8080:8080 chainlit-app:latest


gcloud auth login
gcloud config list

gcloud config set
gcloud config set project PROJECT_ID

gcloud builds submit