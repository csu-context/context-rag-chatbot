# 방화벽 강화 (#133): SSH/UI의 0.0.0.0/0 전체 개방 제거, 백엔드 포트는 VPC 내부로 격리
resource "aws_security_group" "app_sg" {
  name        = "${var.project_name}-app-sg"
  description = "Allow inbound traffic for RAG Chatbot (restricted)"
  vpc_id      = var.vpc_id

  # SSH: 운영팀 IP 대역만 허용 (기본값 빈 목록이면 외부 접근 불가)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidr_blocks
    description = "SSH from approved admin IPs only"
  }

  # Streamlit UI: 허용된 사용자 IP 대역만 접근
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
    description = "Streamlit UI from approved IPs only"
  }

  # 백엔드 포트(ChromaDB 8000, Ollama 11434)는 VPC 내부 통신만 허용
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "ChromaDB - VPC internal only"
  }

  ingress {
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Ollama LLM - VPC internal only"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
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

# 모델명 Single Source of Truth: 레포 .env.example의 MODEL_NAME을 읽어 배포 user_data에 주입한다.
# 하드코딩을 없애 .env.example 한 곳만 바꾸면 EC2 배포까지 동기화된다. 파일/항목이 없으면 앱 기본값으로 폴백. (#181)
locals {
  env_example_path = "${path.module}/../../../.env.example"
  model_name       = try(trimspace(regex("(?m)^MODEL_NAME=(.*)$", file(local.env_example_path))[0]), "gemma4:e2b")
}

resource "aws_instance" "app" {
  ami                  = data.aws_ami.ubuntu_gpu.id
  instance_type        = var.instance_type
  subnet_id            = var.subnet_id
  key_name             = var.key_name
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

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
MODEL_NAME=${local.model_name}
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

echo "Infrastructure & Application Setup Completed!"
EOF

  root_block_device {
    volume_size           = 50
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "${var.project_name}-app-instance"
  }
}

resource "aws_iam_role" "ec2_ecr_role" {
  name = "${var.project_name}-ec2-ecr-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.ec2_ecr_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_ecr_role.name
}

