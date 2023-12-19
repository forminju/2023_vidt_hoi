# 2023 산업공학종합설계 과목 프로젝트 (졸업논문)

## 수정사항
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
