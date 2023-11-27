## 수정사항
<br>1.main 수정완료
<br>2.method-hoi.py 추가
<br>3.matcher.py HOI추가
<br>4.engine.py수정 -done
<br>5.qpic-detr.py vs vidt
<br>(1)def build에서 hoi경우 추가해야됨. -done
<br>(2)vidt-detector.py 에 HOIdetector 추가,build 수정 -done
<br>(3)transformer.py 수정 <- decoder만 사용하도록 -done
<br>-detector.py 530
<br>-transformer.py 53
<br>6.vcoco 변환 -done
<br>7.hoi.py-HOISetcriterion(loss) 수정
<br>-219: outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'} #hoi
<br>-248~ :if 'enc_outputs' in outputs: #hoi
<br>8.hoi.py-HOIdetector-verb_class,sub_obj 레이어 수정
<br>hoi.py-HOIsetcriterion-focal loss 및 one hot으로 수정,def loss_obj_labels(num_interaction->num_boxes)