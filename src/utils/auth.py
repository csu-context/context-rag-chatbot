"""접근 인증 유틸 — 비밀번호는 평문이 아닌 bcrypt 해시로만 다룬다."""

import bcrypt


def verify_password(plain: str, hashed: str | None) -> bool:
    """평문 비밀번호를 bcrypt 해시와 상수시간 비교한다.

    - hashed가 비어 있으면(게이트 미설정) False.
    - 해시 형식이 잘못된 경우에도 예외를 삼키고 False를 반환해(fail-closed) 접근을 거부한다.
    """
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
