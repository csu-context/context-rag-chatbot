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
