## 🔹 사용 데이터셋

프로젝트에서 사용한 데이터셋은 다음과 같습니다: 기본적으로 예시 이미지 하나만 존재하고 나머지 nsfw 이미지 등은 선정성 이유로 삭제했습니다. 

| 데이터셋 | 설명 | 링크 | 참고 논문 |
|-----------|------|------|-----------|
| SYSU-MM01 | RGB-NIR 멀티모달 데이터셋 | [Kaggle](https://www.kaggle.com/datasets/coconutjean/sysumm01) | Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong and Jianhuang Lai. RGB-IR Person Re-Identification by Cross-Modality Similarity Preservation. International Journal of Computer Vision (IJCV), 2020. |
| DeepNIR | NIR-RGB 데이터셋 (`nirscene_img_aug_10_oversample` 포함) | [Kaggle](https://www.kaggle.com/datasets/enddl22/deepnir-nir-rgb-nirscene1-dataset?select=nirscene_img_aug_10_oversample) | - |
| LLVIP | NIR 기반 얼굴 이미지 데이터셋 | [GitHub](https://github.com/bupt-ai-cz/LLVIP) | Jia, X., Zhu, C., Li, M., Tang, W., & Zhou, W. (2021). LLVIP: A visible-infrared paired dataset for low-light vision. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 3496–3504). |

---

## 🔹 다운 필요 모델

| 모델 | 설명 | 링크 |
|------|------|------|
| FalconsAI NSFW Classifier | NSFW 이미지 검출 모델 | [HuggingFace](https://huggingface.co/Falconsai/nsfw_image_detection) |
| MobileSAM | 모바일 환경에서 사용 가능한 Segment Anything 모델 | [GitHub](https://github.com/ChaoningZhang/MobileSAM) |

---

## 🔹 설치 및 환경

```bash
pip install -r requirements.txt

---
## 🔹 추가 사항
현재 train.py에는 ./model 폴더 생성 및 학습 완료된 checkpoint 파일을 .pt 파일로 저장하는 코드가 기본으로 적혀있습니다.
---
