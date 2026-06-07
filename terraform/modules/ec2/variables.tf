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

# 방화벽 강화 (#133): 접근 허용 IP 대역 (0.0.0.0/0 전체 개방 금지)
variable "allowed_cidr_blocks" {
  description = "Streamlit UI 접속 허용 CIDR 대역 (예: [\"1.2.3.0/24\"]). 비어있으면 외부 접근 불가."
  type        = list(string)
  default     = []
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
