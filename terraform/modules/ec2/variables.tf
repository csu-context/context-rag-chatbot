variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "rag-chatbot"
}

variable "key_name" {
  description = "Name of the SSH key pair to use for the instance"
  type        = string
  default     = null
}

# 방화벽 강화 (#133): SSH·백엔드는 IP/VPC로 제한, UI(8501)만 시연 편의로 기본 개방(프로덕션은 tfvars로 제한)
variable "allowed_cidr_blocks" {
  description = "Streamlit UI(8501) 접속 허용 CIDR 대역. 시연 편의를 위해 기본값은 전체 허용이며, 프로덕션 배포 시 tfvars로 제한할 것. (UI 인증 게이트 도입 전까지의 임시 기본값)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_ssh_cidr_blocks" {
  description = "SSH 허용 CIDR 대역 — 운영팀 IP만 등록 (예: [\"10.0.0.0/8\"])"
  type        = list(string)
  default     = []
}

variable "vpc_cidr" {
  description = "VPC CIDR 블록 (백엔드 포트 내부 접근 제한용)"
  type        = string
  default     = "10.0.0.0/16"
}
