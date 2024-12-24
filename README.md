# Recipi app

## Prerequisites
1. Docker
2. Docker compose

## Usage Dockerfile

#### 1. Clone the repository
#### 2. Build the docker image
```bash
docker build -t [image_name] .
```
#### 3. Run the image
```bash
docker run --name {container_name } -p 80:80 -d {image_name}
```

## Usage docker-compose
### 1. Clone the repository
### 2. Run compose file 
```bash
docker compose up -d [--build]
```
