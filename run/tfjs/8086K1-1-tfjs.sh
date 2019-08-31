CARDID=0
TFSCRIPT=v4-BiRnn
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/home/hotdogee/datasets2
DATASET=pfam-regions-d0-s20
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=2
CARDTYPE=TITANV
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2
CHECKPOINT=epoch-6-4951306

export CUDA_VISIBLE_DEVICES=0; tensorflowjs_converter --input_format=tf_saved_model --output_node_names='predictions/ArgMax:0' --saved_model_tags=serve /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/export/1547721584 /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/export/1547721584_web_model

export CUDA_VISIBLE_DEVICES=0; python /home/hotdogee/venv/tf36/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py --input_binary=false --input_graph=/data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/graph.pbtxt --input_checkpoint=/data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/epoch-6-4951306 --output_graph=/data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/frozen-epoch-6-4951306/frozen_graph.pb --output_node_names=predictions/ArgMax:0

export CUDA_VISIBLE_DEVICES=0; tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='predictions/ArgMax' /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/frozen-epoch-6-4951306/frozen_graph.pb /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/frozen-epoch-6-4951306/web_model

# ValueError: Unsupported Ops in the model before optimization
# Iterator, CudnnRNN, IteratorGetNext

# saved_model_cli show --dir /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/export/1547721584 --all
# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['sequences'] tensor_info:
#         dtype: DT_STRING
#         shape: (-1)
#         name: input_example_tensor:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['classes'] tensor_info:
#         dtype: DT_INT32
#         shape: (-1, -1)
#         name: predictions/ArgMax:0
#     outputs['logits'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, -1, 16715)
#         name: output_1/dense/BiasAdd:0
#     outputs['top_classes'] tensor_info:
#         dtype: DT_INT32
#         shape: (-1, -1, 3)
#         name: predictions/TopKV2:1
#     outputs['top_probs'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, -1, 3)
#         name: predictions/TopKV2:0
#   Method name is: tensorflow/serving/predict
