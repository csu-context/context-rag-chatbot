# Terraform State 공유 — S3 + DynamoDB 원격 백엔드로 팀 충돌 방지
# 사용 전 S3 버킷 및 DynamoDB 테이블 사전 생성 필요:
#   aws s3api create-bucket --bucket rag-chatbot-tfstate --region ap-northeast-2 \
#     --create-bucket-configuration LocationConstraint=ap-northeast-2
#   aws s3api put-bucket-versioning --bucket rag-chatbot-tfstate \
#     --versioning-configuration Status=Enabled
#   aws dynamodb create-table --table-name rag-chatbot-tflock \
#     --attribute-definitions AttributeName=LockID,AttributeType=S \
#     --key-schema AttributeName=LockID,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST --region ap-northeast-2

terraform {
  backend "s3" {
    bucket         = "rag-chatbot-tfstate"
    key            = "environments/dev/terraform.tfstate"
    region         = "ap-northeast-2"
    encrypt        = true
    dynamodb_table = "rag-chatbot-tflock"  # 동시 apply 잠금
  }
}