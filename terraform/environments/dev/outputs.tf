output "app_public_ip" {
  description = "Public IP address of the RAG application instance"
  value       = module.ec2.instance_public_ip
}