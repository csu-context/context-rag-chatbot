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

  user_data = <<-EOF
              #!/bin/bash
              # 로그 출력 설정
              exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
              
              echo "Starting Infrastructure Setup..."

              # 1. 시스템 업데이트 및 필수 패키지 설치
              apt-get update -y
              apt-get install -y curl git jq build-essential

              # 2. NVIDIA 드라이버 설치 (g4dn 인스턴스용)
              # Ubuntu 20.04에서 권장되는 드라이버 설치
              apt-get install -y ubuntu-drivers-common
              ubuntu-drivers autoinstall

              # 3. Docker 설치
              curl -fsSL https://get.docker.com -o get-docker.sh
              sh get-docker.sh
              usermod -aG docker ubuntu

              # 4. NVIDIA Container Toolkit 설치 (Docker에서 GPU 사용 필수)
              distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
                && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
              apt-get update
              apt-get install -y nvidia-container-toolkit
              nvidia-ctk runtime configure --runtime=docker
              systemctl restart docker

              # 5. 프로젝트 소스 코드 클론 (실제 환경에서는 본인의 레포지토리 주소로 변경 필요)
              # 여기서는 폴더 구조만 생성하는 예시를 보여줍니다.
              mkdir -p /home/ubuntu/app
              cd /home/ubuntu/app
              
              # 6. Docker Compose 실행
              # 실제로는 여기서 git clone 및 docker-compose up -d 가 실행되어야 합니다.
              # 테스트를 위해 Ollama 컨테이너만 우선 실행하고 모델을 미리 받는 로직을 넣습니다.
              docker run -d --gpus all --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
              
              # Ollama 서비스가 준비될 때까지 대기 후 모델 다운로드
              echo "Waiting for Ollama to start..."
              sleep 30
              docker exec ollama ollama pull llama3.2:1b

              echo "Infrastructure Setup Completed!"
              EOF

  tags = {
    Name = "${var.project_name}-app-instance"
  }
}
