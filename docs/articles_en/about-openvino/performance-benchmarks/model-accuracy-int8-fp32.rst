.. {#openvino_docs_performance_int8_vs_fp32}

Model Accuracy
==============



The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures. The third table presents the GenAI model accuracies as absolute accuracy values. Please also refer to notes below
the table for more information.

* A - Intel® Core™ i9-9000K (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel® Xeon 8480+ (VNNI, AMX), INT8, BF16, FP32
* D - Intel® Flex-170, INT8 and FP16


.. list-table:: Model Accuracy for INT8
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, INT8
     - B, INT8
     - C, INT8
     - D, INT8
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - spearman@cosine
     - 3.33%
     - 3.22%
     - 3.69%
     - 3.28%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.19%
     - 0.06%
     - 0.03%
     - 0.11%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.9%
     - -0.63%
     - -0.61%
     - -0.62%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - -5.64%
     - -0.30%
     - -0.21%
     - -0.28%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.87%
     - -0.87%
     - -0.89%
     - -0.95%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.2%
     - -0.18%
     - -0.18%
     - -0.13%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.03%
     - -0.02%
     - -0.03%
     - -0.0%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - -2.75%
     - -0.11%
     - -0.11%
     - -0.08%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - -6.28%
     - 6.45%
     - 6.46%
     - 6.40%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - -0.30%
     - -0.43%
     - -0.43%
     - -0.87%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - -0.01%
     - -0.04%
     - 0.04%
     - -0.08%

.. list-table:: Model Accuracy for BF16, FP32 and FP16 (FP16: Flex-170 only. BF16: Xeon(R) 8480+ only)
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, FP32
     - B, FP32
     - C, FP32
     - C, BF16
     - D, FP16
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - spearman@cosine
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.03%
     - 0.01%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.04%
     - 0.04%
     - 0.04%
     - 0.06%
     - %
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.02%
     - -0.02%
     - -0.02%
     - -0.02%
     - 0.04%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.01%
     - -0.02%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.18%
     - 0.02%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.01%
     - -0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.02%
     - 0.02%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - 0.01%
     - 0.01%
     - 0.01%
     - 0.04%
     - -0.02%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.03%
     - -0.03%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.25%
     - -0.01%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.04%
     - -0.02%

.. list-table:: Model Accuracy for VNNI-FP16, VNNI-INT8, VNNI-INT4 and MTL-INT4 (Core Ultra)
   :header-rows: 1
   
   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, VNNI-FP16
     - B, VNNI-INT8
     - C, VNNI-INT4
     - D, MTL-INT4
   * - chatGLM2-6b
     - Wikiset
     - ppl
     - 5,24
     - 5.17
     - 6.86
     - 6.87
   * - Falcon-7b-instruct
     - Wikitext
     - ppl
     - 1.65
     - 1.65
     - 1.82
     - 1.82
   * - Llama-2-7b-chat
     - Wikiset
     - ppl
     - 1.54
     - 1.58
     - 1.59
     - 1.59
   * - Mistral-7b
     - Wikitext
     - ppl
     - 1.48
     - 1.48
     - 1.49
     - 1.49
   * - Stable-Diffusion-V2-1
     - LIAON-5B
     - CLIP
     - 
     - 
     - 
     - 

Notes: For all accuracy metrics a "-", (minus sign), indicates an accuracy drop.
For perplexity (ppl) the values do not indicate a deviation from a reference but are the actual measured
accuracy for the model.


.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css">

.. container:: benchmark-banner

   Results may vary. For more information, see
   :doc:`F.A.Q. <./performance-benchmarks-faq>` and
   :doc:`Platforms, Configurations, Methodology <../performance-benchmarks>`.
   See :doc:`Legal Information <../additional-resources/terms-of-use>`.
