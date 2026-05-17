provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source       = "../../modules/vpc"
  project_name = var.project_name
}

module "ec2" {
  source        = "../../modules/ec2"
  vpc_id        = module.vpc.vpc_id
  subnet_id     = module.vpc.public_subnet_id
  instance_type = var.instance_type
  project_name  = var.project_name
}

variable "aws_region" {
  default = "ap-northeast-2"
}

variable "project_name" {
  default = "rag-chatbot-dev"
}

variable "instance_type" {
  default = "g4dn.xlarge"
}

output "app_public_ip" {
  value = module.ec2.instance_public_ip
}
