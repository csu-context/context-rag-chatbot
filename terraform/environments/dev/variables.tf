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
