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

# Issue 45/47: 접근 IP 제한
variable "allowed_cidr_blocks" {
  description = "앱 UI 접속 허용 CIDR (예: [\"YOUR_IP/32\"])"
  type        = list(string)
  default     = []
}

variable "allowed_ssh_cidr_blocks" {
  description = "SSH 허용 CIDR — 운영자 IP만 등록"
  type        = list(string)
  default     = []
}
