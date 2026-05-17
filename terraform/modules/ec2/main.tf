resource "aws_security_group" "app_sg" {
  name        = "${var.project_name}-app-sg"
  description = "Allow inbound traffic for RAG Chatbot"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # 보안을 위해 실제 환경에서는 특정 IP로 제한 필요
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-app-sg"
  }
}

data "aws_ami" "ubuntu_gpu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical (Ubuntu)

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}

resource "aws_instance" "app" {
  ami           = data.aws_ami.ubuntu_gpu.id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id

  vpc_security_group_ids = [aws_security_group.app_sg.id]

  user_data_replace_on_change = true

  user_data = <<-EOF
#!/bin/bash
# 로그 출력 설정
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting Infrastructure Setup..."

# 1. 시스템 업데이트 및 필수 패키지 설치
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl git jq build-essential ca-certificates gnupg lsb-release

# 2. NVIDIA 드라이버 설치 (g4dn 인스턴스용)
echo "Installing NVIDIA Drivers..."
apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall

# 3. Docker 공식 레포지토리 추가 및 설치
echo "Installing Docker..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker
systemctl start docker

# 4. NVIDIA Container Toolkit 설치
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 5. 프로젝트 환경 준비
echo "Setting up project code..."
mkdir -p /home/ubuntu/app
cd /home/ubuntu/app

# 레포지토리 클론 (현재 작업 중인 브랜치 반영을 위해 전체 클론 후 전환)
git clone https://github.com/csu-context/context-rag-chatbot.git .
# 주의: 푸시된 브랜치가 없다면 아래 명령은 실패할 수 있으나 기본 브랜치로 동작함
git checkout feature/issue-89-local-infra || echo "Branch not found, using default"

# 6. .env 파일 자동 생성
echo "Creating .env file..."
cat <<EOT > .env
MODEL_TYPE=ollama
MODEL_NAME=llama3.2:1b
EMBEDDING_MODEL_NAME=BAAI/bge-m3
OLLAMA_BASE_URL=http://ollama:11434
CHROMA_SERVER_HOST=chromadb
CHROMA_SERVER_PORT=8000
PARSER_TYPE=manual
RETRIEVER_TYPE=hybrid
DATA_PATH=./data
DB_PATH=./vector_db
EOT

chown -R ubuntu:ubuntu /home/ubuntu/app

# 7. Docker Compose 실행 (전체 서비스 기동)
echo "Starting services with docker-compose..."
docker compose up -d

# 8. Ollama 모델 사전 로드 (백그라운드에서 진행)
echo "Pulling Ollama model in background..."
docker exec rag-chatbot-ollama ollama pull llama3.2:1b &

echo "Infrastructure & Application Setup Completed!"
EOF

  tags = {
    Name = "${var.project_name}-app-instance"
  }
}
