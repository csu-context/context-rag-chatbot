output "app_public_ip" {
  description = "Public IP address of the RAG application instance"
  value       = module.ec2.instance_public_ip
}

# Issue 24: Elastic IP 출력
output "app_elastic_ip" {
  description = "Elastic IP (고정) — 이 주소로 서비스에 접근하세요"
  value       = module.ec2.elastic_ip
}