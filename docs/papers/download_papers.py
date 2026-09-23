"""Download the open-access papers cited by the contest document.

Run from the repository root: python docs/papers/download_papers.py
The script validates each response as a PDF and writes a local manifest.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


DEST = Path(__file__).resolve().parent
PAPERS = [
    ("02_Beyond_Missing_Modalities_CVPR2026.pdf", "https://openaccess.thecvf.com/content/CVPR2026/papers/Qiu_Beyond_Missing_Modalities_Hypergraph_Conditioned_Diffusion_for_Uncertainty-Aware_Multimodal_Emotion_CVPR_2026_paper.pdf"),
    ("04_CMAD_ICCV2025.pdf", "https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf"),
    ("05_Proxy_Driven_Robust_MSA_ACL2025.pdf", "https://aclanthology.org/2025.acl-long.1075.pdf"),
    ("06_Learning_Invariant_Modality_Representation_ACL2026.pdf", "https://aclanthology.org/2026.acl-long.2119.pdf"),
    ("07_EMOE_CVPR2025.pdf", "https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf"),
    ("08_Locate_and_Explain_ACL2026.pdf", "https://aclanthology.org/2026.acl-long.2012.pdf"),
    ("09_CaReFlow_CVPR2026.pdf", "https://openaccess.thecvf.com/content/CVPR2026/papers/Mai_CaReFlow_Cyclic_Adaptive_Rectified_Flow_for_Multimodal_Fusion_CVPR_2026_paper.pdf"),
    ("10_AUMDF_CJC2025.pdf", "https://lib.zjsru.edu.cn/25-10.11-4.pdf"),
    ("11_SlotSPE_ICLR2026.pdf", "https://arxiv.org/pdf/2512.01116"),
]


def main() -> None:
    records = []
    for filename, url in PAPERS:
        target = DEST / filename
        if target.exists() and target.read_bytes().startswith(b"%PDF-"):
            data = target.read_bytes()
            status = "already_present"
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MOSEI-ARE literature archive)"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-4096:]:
                    raise ValueError("Response is not a complete PDF")
                target.write_bytes(data)
                status = "downloaded"
            except (OSError, urllib.error.URLError, ValueError) as exc:
                print(f"FAILED {filename}: {exc}")
                records.append({"file": filename, "url": url, "status": "failed", "error": str(exc)})
                continue
        records.append({"file": filename, "url": url, "status": status,
                        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(f"OK {filename} ({len(data):,} bytes)")
    (DEST / "download_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
