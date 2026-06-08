variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "ap-northeast-2"
}

variable "project_name" {
  description = "Project name tag"
  type        = string
  default     = "rag-chatbot-dev"
}

variable "instance_type" {
  description = "EC2 instance type for GPU support"
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "AWS Key Pair name"
  type        = string
  default     = "rag-chatbot-key"
}

# 방화벽 강화 (#133): SSH는 운영자 IP로 제한, UI(8501)는 시연 위해 기본 개방
variable "allowed_cidr_blocks" {
  description = "앱 UI(8501) 접속 허용 CIDR. 시연 편의로 기본값은 전체 허용, 프로덕션 배포 시 tfvars로 제한."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_ssh_cidr_blocks" {
  description = "SSH 허용 CIDR — 운영자 IP만 등록"
  type        = list(string)
  default     = []
}
