# Issue 45/47: 방화벽 취약 — SSH/앱 포트 0.0.0.0/0 전체 개방 제거
# var.allowed_cidr_blocks 로 허용 IP 대역 제한 (기본값: 비어있어 접근 불가)
resource "aws_security_group" "app_sg" {
  name        = "${var.project_name}-app-sg"
  description = "Allow inbound traffic for RAG Chatbot (restricted)"
  vpc_id      = var.vpc_id

  # SSH: 운영팀 IP 대역만 허용
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

  # Issue 50: 백엔드 포트(ChromaDB 8000, Ollama 11434)는 내부 VPC 통신만 허용
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "ChromaDB — VPC internal only"
  }

  ingress {
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Ollama LLM — VPC internal only"
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

# Issue 48: EC2 인스턴스 프로파일 — SSM 파라미터 읽기 권한
resource "aws_iam_role" "app_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ssm_read" {
  name = "${var.project_name}-ssm-read"
  role = aws_iam_role.app_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
      Resource = "arn:aws:ssm:*:*:parameter${var.ssm_parameter_prefix}/*"
    }]
  })
}

resource "aws_iam_instance_profile" "app_profile" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.app_role.name
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
  key_name      = var.key_name

  vpc_security_group_ids = [aws_security_group.app_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.app_profile.name

  user_data_replace_on_change = true

  # Issue 48: Secret 하드코딩 제거 — SSM Parameter Store에서 런타임 조회
  user_data = <<-EOF
#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting Infrastructure Setup..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl git jq build-essential ca-certificates gnupg lsb-release awscli

# NVIDIA 드라이버
apt-get install -y ubuntu-drivers-common && ubuntu-drivers autoinstall

# Docker
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable docker && systemctl start docker

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker

# 프로젝트 코드
mkdir -p /home/ubuntu/app && cd /home/ubuntu/app
git clone https://github.com/csu-context/context-rag-chatbot.git .
git checkout develop || echo "Using default branch"

# Issue 48: .env 생성 — 민감 정보는 SSM Parameter Store에서 런타임 조회
echo "Fetching secrets from SSM Parameter Store..."
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
PREFIX="${var.ssm_parameter_prefix}"

get_ssm() {
  aws ssm get-parameter --name "$PREFIX/$1" --with-decryption \
    --region "$REGION" --query "Parameter.Value" --output text 2>/dev/null || echo ""
}

cat <<EOT > .env
MODEL_TYPE=ollama
MODEL_NAME=llama3.2:1b
EMBEDDING_MODEL_NAME=BAAI/bge-m3
OLLAMA_BASE_URL=http://ollama:11434
CHROMA_SERVER_HOST=chromadb
CHROMA_SERVER_PORT=8000
PARSER_TYPE=manual
RETRIEVER_TYPE=hybrid
ANTHROPIC_API_KEY=$(get_ssm "ANTHROPIC_API_KEY")
GOOGLE_API_KEY=$(get_ssm "GOOGLE_API_KEY")
APP_PASSWORD=$(get_ssm "APP_PASSWORD")
ADMIN_PASSWORD=$(get_ssm "ADMIN_PASSWORD")
EOT

chown -R ubuntu:ubuntu /home/ubuntu/app
docker compose up -d
echo "Setup Completed!"
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
