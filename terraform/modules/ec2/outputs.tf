output "instance_public_ip" {
  value = aws_instance.app.public_ip
}

# Issue 24: Elastic IP 출력 — 고정 IP로 접근 가능
output "elastic_ip" {
  description = "Elastic IP (고정 IP) — 재시작 후에도 동일한 주소 사용"
  value       = aws_eip.app.public_ip
}