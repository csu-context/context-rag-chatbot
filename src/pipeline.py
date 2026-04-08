import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.common.constants import MetadataFields
from src.data.parser import ManualParser
from src.processing.chunking import create_parent_child_chunks, split_into_children
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        """
        통합 전처리 파이프라인 (#18 이슈 대응)
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        ensure_directories()

    def run(self, save_filename: str = None) -> List[Dict[str, Any]]:
        """
        전체 전처리 파이프라인 실행: 스캔 -> (파싱+표준화) -> 계층적 청킹 -> 저장
        """
        all_hierarchical_data = []
        
        # 1. 파일 목록 스캔
        supported_exts = [".pdf", ".md", ".markdown"]
        files_to_process = []
        for ext in supported_exts:
            files_to_process.extend(list(self.raw_dir.glob(f"**/*{ext}")))
        
        if not files_to_process:
            logger.warning(f"처리할 파일을 찾을 수 없습니다: {self.raw_dir}")
            return []

        logger.info(f"총 {len(files_to_process)}개의 파일에 대해 전처리를 시작합니다.")

        # tqdm으로 진행률 표시
        for file_path in tqdm(files_to_process, desc="Preprocessing Files"):
            relative_path = file_path.relative_to(self.raw_dir)
            
            try:
                # 파서 초기화 (여기서 source_id 등이 자동으로 생성됨)
                parser = ManualParser(str(relative_path))
                
                if parser.extension == "pdf":
                    # PDF 파싱 (표준 규격 메타데이터 포함)
                    sections = parser.parse()
                    
                    # PDF 섹션들을 계층 구조로 변환
                    for sec in sections:
                        parent_id = str(uuid.uuid4())
                        
                        # [상수 적용] MetadataFields 사용
                        meta_for_children = sec["metadata"].copy()
                        meta_for_children[MetadataFields.SEC_TITLE] = f"{sec['chapter']} > {sec['article']}"
                        
                        children = split_into_children(sec["content"], parent_id, meta_for_children)
                            
                        all_hierarchical_data.append({
                            MetadataFields.PARENT_ID: parent_id,
                            "parent_text": sec["content"],
                            "metadata": sec["metadata"],
                            "children": children
                        })
                else:
                    # Markdown 처리 (표준화된 create_parent_child_chunks 호출)
                    with open(file_path, "r", encoding="utf-8") as f:
                        md_text = f.read()
                    
                    # 파서에서 생성된 기본 메타데이터 전달 (상수 키 사용)
                    base_metadata = {
                        MetadataFields.SOURCE_ID: parser.source_id,
                        MetadataFields.SRC_NAME: parser.file_name,
                        MetadataFields.DOC_TYPE: parser.extension,
                        MetadataFields.PG_NUM: 1
                    }
                    file_chunks = create_parent_child_chunks(md_text, base_metadata)
                    all_hierarchical_data.extend(file_chunks)
                    
            except Exception as e:
                logger.error(f"실패: {relative_path} - {str(e)}")

        # 2. 결과 저장
        if all_hierarchical_data:
            if not save_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"preprocessed_v1_{timestamp}.json"
            
            save_path = self.processed_dir / save_filename
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_hierarchical_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"성공: 전처리 완료 ({len(all_hierarchical_data)}개 섹션) -> {save_path}")
        
        return all_hierarchical_data

if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
