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
  key_name      = var.key_name

  # Issue 45/47/48/50: 보안 강화 파라미터
  allowed_cidr_blocks     = var.allowed_cidr_blocks
  allowed_ssh_cidr_blocks = var.allowed_ssh_cidr_blocks
  ssm_parameter_prefix    = "/rag-chatbot/dev"
  vpc_cidr                = module.vpc.vpc_cidr
}
