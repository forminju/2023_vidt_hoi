# 2023 산업공학종합설계 과목 프로젝트 (졸업논문)

## Abstract
  이 논문은 인간과 객체 간의 상호작용(Human-Object Interaction, HOI) 탐지를 위한 새로운
접근 방식을 제안합니다. 연구의 주된 목적은 HOI 탐지 분야에 기존 모델들의 한계를
극복하고, 성능을 향상시키는 새로운 모델 구조를 개발하는 것입니다. 이를 위해, 우리는
Vision and Detection Transformers(ViDT) [1] 와 Query-based Pairwise Interaction Classifier(QPIC)
[2]를 결합하여, DETR(End-to-End Object Detection with Transformers) 기반 모델의 단점을
해결하고자 했습니다.
ViDT [1]는 기존의 DETR 모델을 대체하는 새로운 객체 탐지 기반 모델로, Swin Transformer의
Attention 모델을 재구성하고 경량화된 구조를 통해 계산 부담을 줄이는 동시에 검출 성능과
속도를 향상시켰습니다. 이 연구에서는 ViDT를 통해, DETR 모델의 긴 학습 시간, 복잡한
최적화 과정, 그리고 긴 추론 시간이라는 주요한 제약을 극복하고, HOI 탐지의 전반적인
성능을 개선하고자 했습니다.
연구 결과, ViDT와 QPIC을 결합한 새로운 모델은 기대한 바에 미치지 못했습니다. 실험은
V-COCO 데이터셋을 활용하여 진행되었으며, 주요 평가 지표로는 평균 정밀도(mAP)가
사용되었습니다. 실험 결과는 여러 한계점을 드러냈으며, 이는 향후 연구 방향을 제시하는
중요한 지표로 활용될 수 있습니다.

## 수정사항
### Main branch:Anchor Embedding X
<br>1.main.py- if arf.hoi 추가
<br>2.method-vidt-hoi.py(from QPIC) 추가
<br>3.matcher.py-def HungarianMatcherHOI 추가
<br>4.engine.py-def evaluate_hoi 추가
<br>5.method-vidt-detector.py-기존 object detection에 사용되는 class Detector를 수정한 class HOIDetector 추가
<br> -328 __init__에 num_obj_classes, num_verb_classes 변수추가
<br> -357 self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
<br> -359 self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
 
<br> -599 out = {'pred_obj_logits': outputs_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_coord[-1]}
 <br> ->def forward의 아웃풋으로 pred_verb_logits(interaction),pred_sub_boxes을 반환하게 수정 #hoi참조

<br> -def build - if args.hoi 추가
<br>6.datasets-hico.py,hico_eval.py,vcoco.py,vcoco_eval.py 추가
<br>7.hoi.py-HOISetcriterion(loss) vidt에 맞게 수정
<br> -110~
  <br>target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], 
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        <br>target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        <br>target_classes_onehot = target_classes_onehot[:,:,:-1]

        <br>loss_obj_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]

<br>-> def loss_obj_labels을 focal loss 및 one hot class tensor를 반환하게 수정,(num_interaction->num_boxes)

<br> -229: outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}로 수정
<br> -246~: if 'aux_outputs' in outputs: 에서 num_boxes -> num_interaction으로 수정
<br> -258~ :if 'enc_outputs' in outputs: 추가 및 num_boxes -> num_interaction으로 수정
<br>9.arguments.py #HOI arguments,'--set_cost_verb_class','--obj_loss_coef','--verb_loss_coef','--hoi_path','--num_queries' arguments 추가
### test1 branch:Anchor Embedding O

hoi.py-HOIdetector-verb_class,sub_obj 레이어 수정
hoi.py-HOIsetcriterion-focal loss 및 one hot으로 수정,def loss_obj_labels(num_interaction->num_boxes)
QAHOI_Anchor embedding 추가-detector.py-HOIDetector 536~549

