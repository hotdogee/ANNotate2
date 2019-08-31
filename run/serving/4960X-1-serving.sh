CARDID=0
export CUDA_VISIBLE_DEVICES=0; docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/home/hotdogee/export,target=/models/pfam -e MODEL_NAME=pfam -t tensorflow/serving:latest-gpu

# export CUDA_VISIBLE_DEVICES=${CARDID}; tensorflow_model_server --rest_api_port=8501 --model_name=pfam --model_base_path="/data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/export"

# export CUDA_VISIBLE_DEVICES=0; tensorflow_model_server --rest_api_port=8501 --model_name=pfam --model_base_path="/home/hotdogee/export"

# cp -r /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn/RewarmupGru512x4_step1447452_b1_ws0_lr0.0065_TITANV_8086K1-1.2/export /home/hotdogee/

# docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/home/hotdogee/export,target=/models/pfam -e MODEL_NAME=pfam -t tensorflow/serving:latest-gpu

# ["EiIKIAoHcHJvdGVpbhIVCgUKAwoBAQoFCgMKAQIKBQoDCgED", "EikKJwoHcHJvdGVpbhIcCgUKAwoBBAoFCgMKAQUKBQoDCgEBCgUKAwoBBg==", "EhsKGQoHcHJvdGVpbhIOCgUKAwoBAQoFCgMKAQI="]


curl -d '{"instances": ["FLIM","VP","AWGST"]}' -X POST http://localhost:8501/v1/models/pfam:predict
curl -d '{"instances": ["FLIM","VP","AWGST"]}' -X POST http://192.168.1.63:8501/v1/models/pfam:predict
# {
#     "predictions": [
#         {
#             "classes": [1, 1, 1, 1, 1],
#             "top_probs": [[0.863652, 0.00359598, 0.0030836], [0.872175, 0.00586935, 0.00373591], [0.922818, 0.00463576, 0.00420858], [0.986909, 0.00216449, 0.000734239], [0.999964, 3.24868e-05, 3.34089e-06]],
#             "top_classes": [[1, 270, 9580], [1, 143, 371], [1, 143, 371], [1, 371, 661], [1, 371, 273]]
#         },
#         {
#             "classes": [1, 1, 1, 1, 1],
#             "top_probs": [[0.955396, 0.00675229, 0.00196275], [0.992561, 0.00175544, 0.00112648], [0.998756, 0.00117824, 5.98112e-05], [0.999878, 0.000121194, 3.58333e-07], [0.999115, 0.000884879, 1.94156e-08]],
#             "top_classes": [[1, 12957, 499], [1, 371, 499], [1, 371, 273], [1, 273, 371], [1, 273, 295]]
#         },
#         {
#             "classes": [1, 1, 1, 1, 1],
#             "top_probs": [[0.99927, 7.11499e-05, 3.98093e-05], [0.999686, 2.90164e-05, 2.85827e-05], [0.999923, 4.07744e-05, 9.31633e-06], [0.999945, 4.71734e-05, 5.4188e-06], [0.999961, 3.62744e-05, 2.88229e-06]],
#             "top_classes": [[1, 13607, 3581], [1, 13607, 273], [1, 273, 503], [1, 273, 295], [1, 273, 295]]
#         }
#     ]
# }

curl -d '{"instances": [{"b64":"EiIKIAoHcHJvdGVpbhIVCgUKAwoBAQoFCgMKAQIKBQoDCgED"}, {"b64":"EikKJwoHcHJvdGVpbhIcCgUKAwoBBAoFCgMKAQUKBQoDCgEBCgUKAwoBBg=="}, {"b64":"EhsKGQoHcHJvdGVpbhIOCgUKAwoBAQoFCgMKAQI="}]}' -X POST http://localhost:8501/v1/models/pfam:predict
# {
#     "predictions": [
#         {
#             "top_probs": [[0.908537, 0.00903837, 0.00292493], [0.920916, 0.0127167, 0.0064935], [0.982364, 0.00733917, 0.00133529], [0.999859, 0.000135791, 5.45226e-06]],
#             "top_classes": [[1, 371, 143], [1, 371, 143], [1, 371, 143], [1, 371, 273]],
#             "classes": [1, 1, 1, 1]
#         },
#         {
#             "top_probs": [[0.99979, 1.21808e-05, 8.74761e-06], [0.999963, 2.22366e-06, 1.66441e-06], [0.999997, 5.23413e-07, 5.06091e-07], [1.0, 4.9044e-08, 2.88793e-08]],
#             "top_classes": [[1, 270, 11992], [1, 203, 13747], [1, 203, 371], [1, 273, 371]],
#             "classes": [1, 1, 1, 1]
#         },
#         {
#             "top_probs": [[0.969021, 0.00418757, 0.0015526], [0.982476, 0.00132226, 0.000599342], [0.999913, 8.3299e-05, 2.49705e-06], [0.999995, 4.89479e-06, 8.11999e-08]],
#             "top_classes": [[1, 9580, 371], [1, 204, 9580], [1, 371, 41], [1, 273, 371]],
#             "classes": [1, 1, 1, 1]
#         }
#     ]
# }

# curl http://localhost:8501/v1/models/pfam
# {
#  "model_version_status": [
#   {
#    "version": "1547721584",
#    "state": "AVAILABLE",
#    "status": {
#     "error_code": "OK",
#     "error_message": ""
#    }
#   }
#  ]
# }

# curl http://localhost:8501/v1/models/pfam/metadata
# {
# "model_spec":{
#  "name": "pfam",
#  "signature_name": "",
#  "version": "1547721584"
# }
# ,
# "metadata": {"signature_def": {
#  "signature_def": {
#   "serving_default": {
#    "inputs": {
#     "sequences": {
#      "dtype": "DT_STRING",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "input_example_tensor:0"
#     }
#    },
#    "outputs": {
#     "classes": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "predictions/ArgMax:0"
#     },
#     "logits": {
#      "dtype": "DT_FLOAT",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "16715",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "output_1/dense/BiasAdd:0"
#     },
#     "top_probs": {
#      "dtype": "DT_FLOAT",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "3",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "predictions/TopKV2:0"
#     },
#     "top_classes": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "3",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "predictions/TopKV2:1"
#     }
#    },
#    "method_name": "tensorflow/serving/predict"
#   }
#  }
# }
# }
# }
curl -d '{"instances": ["MAKYIFVTGGVVSSLGKGITAASLGRLLKNRGVKVTTQKFDPYINVDPGTMSPYQHGEVFVTDDGAETDLDLGHYERFIDINLNKFSSVTTGKIYSTVLRKERRGEYLGGTVQVIPHVTNEIKERVFRAGRETNADVVITEIGGTVGDIESLPFLEAIRQIKSDVGSENVMYIHCTLIPYIKAAGELKTKPTQHSVKELRSLGIQPNIIVVRTEMPVSDEMKEKLGLFCDINPKAVIEARDADTLYQVPIALQEQNMDQLVCDHLKLDCKEADMSEWKGLIDRVLNLSGKTKIALVGKYVALQDAYISVVESLRHAGYHHDADIEIDWINSEEVTEENVVERLKGADGILVPGGFGDRGVDGKLLAIQYARENNIPYLGICLGMQLASIEFARNVLGYKDAHSTELKPSTKYPIIDLLPEQKDVEDLGGTLRLGLYPCKLSENTKAYEAYNDEVVYERHRHRYEFNNEFREQMEAKGFIFSGTSPDGRLVEIVELQDHPWFVAAQFHPEFISRPTRPQPLFRDFIGATLQLQK"]}' -X POST https://ann.hanl.in/v1/models/pfam:predict

curl -d '{"instances": ["MAKYIFVTGGVVSSLGKGITAASLGRLLKNRGVKVTTQKFDPYINVDPGTMSPYQHGEVFVTDDGAETDLDLGHYERFIDINLNKFSSVTTGKIYSTVLRKERRGEYLGGTVQVIPHVTNEIKERVFRAGRETNADVVITEIGGTVGDIESLPFLEAIRQIKSDVGSENVMYIHCTLIPYIKAAGELKTKPTQHSVKELRSLGIQPNIIVVRTEMPVSDEMKEKLGLFCDINPKAVIEARDADTLYQVPIALQEQNMDQLVCDHLKLDCKEADMSEWKGLIDRVLNLSGKTKIALVGKYVALQDAYISVVESLRHAGYHHDADIEIDWINSEEVTEENVVERLKGADGILVPGGFGDRGVDGKLLAIQYARENNIPYLGICLGMQLASIEFARNVLGYKDAHSTELKPSTKYPIIDLLPEQKDVEDLGGTLRLGLYPCKLSENTKAYEAYNDEVVYERHRHRYEFNNEFREQMEAKGFIFSGTSPDGRLVEIVELQDHPWFVAAQFHPEFISRPTRPQPLFRDFIGATLQLQK"]}' -X POST http://localhost:8501/v1/models/pfam:predict

# {
#     "predictions": [
#         {
#             "classes": [1, 1, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1185, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 1, 1, 1, 1, 1, 1, 1],
#             "top_probs": [[0.999283, 0.00069552, 1.40857e-05], [0.971113, 0.0288162, 2.96982e-05], [0.999206, 0.000793066, 2.57342e-07], [0.999918, 8.06849e-05, 4.59388e-07], [0.999965, 3.34391e-05, 8.84439e-07], [0.999996, 3.32527e-06, 2.95344e-07], [0.999979, 1.99964e-05, 8.85776e-07], [0.999997, 2.36212e-06, 2.69686e-07], [0.999998, 1.7161e-06, 7.19072e-08], [0.999999, 1.08918e-06, 3.59931e-08], [0.999999, 6.98104e-07, 2.28395e-08], [0.999997, 2.49814e-06, 9.16156e-08], [0.999999, 1.09121e-06, 6.48425e-08], [0.999999, 8.53107e-07, 5.82445e-08], [0.999997, 2.90527e-06, 1.09194e-07], [0.999998, 1.48759e-06, 5.62243e-07], [0.999998, 9.26225e-07, 8.53233e-07], [0.999996, 3.12129e-06, 5.18634e-07], [0.999997, 2.66359e-06, 7.9408e-07], [0.999998, 1.03245e-06, 4.94861e-07], [0.999997, 2.11585e-06, 4.35322e-07], [0.999997, 2.15692e-06, 3.51787e-07], [0.999999, 5.47055e-07, 3.61838e-07], [0.999999, 8.24402e-07, 4.6179e-07], [0.999998, 1.15028e-06, 3.71946e-07], [0.999999, 3.75457e-07, 3.41905e-07], [0.999999, 5.31317e-07, 3.46036e-07], [0.999999, 4.76481e-07, 3.74016e-07], [0.999999, 3.52842e-07, 1.879e-07], [0.999999, 3.29314e-07, 2.83757e-07], [1.0, 2.95199e-07, 1.22177e-07], [1.0, 2.62121e-07, 2.48216e-07], [0.999999, 3.20591e-07, 2.93675e-07], [1.0, 2.58424e-07, 1.01145e-07], [1.0, 2.7061e-07, 1.50831e-07], [1.0, 2.44998e-07, 7.18229e-08], [1.0, 2.20315e-07, 1.07604e-07], [1.0, 1.81113e-07, 1.07203e-07], [1.0, 1.8535e-07, 6.6569e-08], [1.0, 2.14872e-07, 8.20802e-08], [1.0, 1.777e-07, 6.09892e-08], [1.0, 2.1202e-07, 4.60927e-08], [1.0, 1.96953e-07, 6.50098e-08], [1.0, 1.78086e-07, 5.15174e-08], [1.0, 1.47293e-07, 3.46943e-08], [1.0, 1.7366e-07, 3.16606e-08], [1.0, 1.60908e-07, 2.29671e-08], [1.0, 1.53497e-07, 1.42033e-08], [1.0, 1.75758e-07, 1.52447e-08], [1.0, 1.75936e-07, 1.73019e-08], [1.0, 2.57656e-07, 1.84941e-08], [1.0, 1.99096e-07, 1.61505e-08], [1.0, 1.60549e-07, 1.14692e-08], [1.0, 1.97757e-07, 1.12156e-08], [1.0, 1.71221e-07, 9.19847e-09], [1.0, 1.50275e-07, 6.31196e-09], [1.0, 1.37477e-07, 5.14223e-09], [1.0, 1.18456e-07, 5.85031e-09], [1.0, 1.5494e-07, 5.40011e-09], [1.0, 1.35885e-07, 4.33514e-09], [1.0, 1.77481e-07, 5.11725e-09], [1.0, 1.41498e-07, 4.35442e-09], [1.0, 1.23822e-07, 4.1546e-09], [1.0, 9.68066e-08, 3.6846e-09], [1.0, 7.61726e-08, 3.03729e-09], [1.0, 8.4065e-08, 3.46959e-09], [1.0, 8.70321e-08, 4.65484e-09], [1.0, 9.29975e-08, 4.03552e-09], [1.0, 8.70926e-08, 5.01829e-09], [1.0, 1.04153e-07, 4.3144e-09], [1.0, 8.76306e-08, 3.86325e-09], [1.0, 8.38124e-08, 3.33136e-09], [1.0, 7.58618e-08, 3.93762e-09], [1.0, 5.94604e-08, 3.10972e-09], [1.0, 4.25381e-08, 1.83715e-09], [1.0, 3.5992e-08, 1.64467e-09], [1.0, 2.87007e-08, 1.23778e-09], [1.0, 2.83395e-08, 9.424e-10], [1.0, 3.85542e-08, 1.14455e-09], [1.0, 3.33608e-08, 1.20613e-09], [1.0, 3.12682e-08, 9.31997e-10], [1.0, 3.94707e-08, 1.48233e-09], [1.0, 3.53108e-08, 1.12627e-09], [1.0, 3.94591e-08, 1.38676e-09], [1.0, 4.13962e-08, 1.62244e-09], [1.0, 4.04989e-08, 1.14683e-09], [1.0, 3.33632e-08, 1.11011e-09], [1.0, 3.67484e-08, 1.52282e-09], [1.0, 4.38619e-08, 1.58753e-09], [1.0, 3.649e-08, 1.77061e-09], [1.0, 3.79999e-08, 2.31823e-09], [1.0, 2.95941e-08, 1.72142e-09], [1.0, 2.6055e-08, 1.9897e-09], [1.0, 2.14871e-08, 1.18252e-09], [1.0, 2.07518e-08, 9.14291e-10], [1.0, 1.71625e-08, 9.83154e-10], [1.0, 1.6161e-08, 9.32731e-10], [1.0, 1.46162e-08, 5.7279e-10], [1.0, 1.72293e-08, 5.37841e-10], [1.0, 2.16274e-08, 1.05627e-09], [1.0, 1.70732e-08, 6.76511e-10], [1.0, 1.72293e-08, 8.33595e-10], [1.0, 1.52724e-08, 7.19682e-10], [1.0, 1.68707e-08, 1.07273e-09], [1.0, 1.26889e-08, 6.42862e-10], [1.0, 1.32034e-08, 6.17391e-10], [1.0, 1.29446e-08, 4.69008e-10], [1.0, 1.33031e-08, 3.50845e-10], [1.0, 1.20363e-08, 4.31798e-10], [1.0, 1.09566e-08, 4.65183e-10], [1.0, 8.33266e-09, 4.09818e-10], [1.0, 1.07633e-08, 4.90158e-10], [1.0, 9.38551e-09, 5.61299e-10], [1.0, 8.13438e-09, 4.57055e-10], [1.0, 1.51631e-08, 8.05816e-10], [1.0, 1.36196e-08, 9.27926e-10], [1.0, 1.05402e-08, 6.75604e-10], [1.0, 1.11655e-08, 7.46737e-10], [1.0, 9.34309e-09, 7.10128e-10], [1.0, 7.4447e-09, 5.74538e-10], [1.0, 6.47537e-09, 4.54845e-10], [1.0, 5.02652e-09, 4.19167e-10], [1.0, 5.44288e-09, 4.3695e-10], [1.0, 3.97737e-09, 3.3787e-10], [1.0, 4.22953e-09, 3.29905e-10], [1.0, 3.40433e-09, 2.68111e-10], [1.0, 3.62824e-09, 2.72862e-10], [1.0, 4.29787e-09, 3.09949e-10], [1.0, 3.7255e-09, 2.51338e-10], [1.0, 2.92459e-09, 1.86085e-10], [1.0, 2.30406e-09, 1.41631e-10], [1.0, 2.0543e-09, 1.23427e-10], [1.0, 1.70962e-09, 1.06142e-10], [1.0, 1.61171e-09, 1.0812e-10], [1.0, 1.14479e-09, 9.10577e-11], [1.0, 1.10455e-09, 9.62196e-11], [1.0, 9.57396e-10, 9.93724e-11], [1.0, 9.74666e-10, 1.22822e-10], [1.0, 1.03029e-09, 1.5152e-10], [1.0, 9.86872e-10, 1.76203e-10], [1.0, 1.22183e-09, 2.57852e-10], [1.0, 1.03957e-09, 2.93586e-10], [1.0, 3.04715e-09, 8.55443e-10], [1.0, 3.53419e-09, 1.22407e-09], [1.0, 4.25667e-09, 1.53765e-09], [1.0, 5.57903e-09, 1.63785e-09], [1.0, 5.81216e-09, 1.542e-09], [1.0, 6.58963e-09, 1.66566e-09], [1.0, 8.25326e-09, 1.90085e-09], [1.0, 6.44006e-09, 1.47563e-09], [1.0, 6.64121e-09, 1.67066e-09], [1.0, 7.8989e-09, 1.84805e-09], [1.0, 7.25292e-09, 1.45804e-09], [1.0, 5.58008e-09, 1.47597e-09], [1.0, 5.74375e-09, 1.32279e-09], [1.0, 5.28227e-09, 1.0955e-09], [1.0, 4.53587e-09, 1.176e-09], [1.0, 5.22786e-09, 1.1105e-09], [1.0, 4.82968e-09, 8.04069e-10], [1.0, 6.1966e-09, 8.37127e-10], [1.0, 6.9143e-09, 9.31078e-10], [1.0, 6.85344e-09, 7.24511e-10], [1.0, 7.05888e-09, 7.49511e-10], [1.0, 7.51369e-09, 7.52335e-10], [1.0, 7.81592e-09, 8.62955e-10], [1.0, 7.85379e-09, 7.42914e-10], [1.0, 7.0115e-09, 7.27341e-10], [1.0, 6.57053e-09, 7.49542e-10], [1.0, 6.07835e-09, 6.53751e-10], [1.0, 4.85305e-09, 6.36347e-10], [1.0, 4.55601e-09, 5.79912e-10], [1.0, 4.05638e-09, 5.06505e-10], [1.0, 4.15181e-09, 6.01245e-10], [1.0, 3.72163e-09, 5.68158e-10], [1.0, 3.88166e-09, 7.60813e-10], [1.0, 4.31189e-09, 7.27246e-10], [1.0, 4.59948e-09, 1.0085e-09], [1.0, 6.79172e-09, 8.8734e-10], [1.0, 8.06243e-09, 6.79402e-10], [1.0, 1.05132e-08, 1.47497e-09], [1.0, 1.37798e-08, 1.00071e-09], [1.0, 1.68175e-08, 4.83994e-10], [1.0, 2.08057e-08, 5.28494e-10], [1.0, 2.3804e-08, 4.96649e-10], [1.0, 2.71289e-08, 5.4361e-10], [1.0, 2.37138e-08, 4.00718e-10], [1.0, 2.3974e-08, 5.90259e-10], [1.0, 1.73291e-08, 2.26026e-10], [1.0, 1.33195e-08, 2.46919e-10], [1.0, 1.0979e-08, 1.31952e-10], [1.0, 6.45131e-09, 8.08465e-11], [1.0, 5.18759e-09, 7.82449e-11], [1.0, 3.4961e-09, 4.46158e-11], [1.0, 2.69828e-09, 4.26302e-11], [1.0, 1.77559e-09, 3.63215e-11], [1.0, 1.65891e-09, 4.14949e-11], [1.0, 1.19197e-09, 3.60768e-11], [1.0, 9.22079e-10, 3.54796e-11], [1.0, 6.22579e-10, 2.35559e-11], [1.0, 5.54826e-10, 2.10078e-11], [1.0, 4.69592e-10, 1.88079e-11], [1.0, 3.42772e-10, 1.33289e-11], [1.0, 3.28827e-10, 9.52009e-12], [1.0, 3.61891e-10, 2.54568e-11], [1.0, 3.05579e-10, 1.65467e-11], [1.0, 4.01542e-10, 3.04286e-11], [1.0, 4.92341e-10, 2.85968e-11], [1.0, 9.38728e-10, 1.75888e-10], [1.0, 1.35924e-09, 1.75886e-10], [1.0, 2.82297e-09, 2.64379e-10], [1.0, 1.16729e-08, 1.45287e-09], [1.0, 1.90681e-08, 7.17311e-10], [1.0, 4.13263e-08, 2.27235e-09], [1.0, 1.81108e-07, 3.76222e-09], [1.0, 1.64986e-07, 1.28564e-08], [0.999999, 6.26001e-07, 7.92963e-09], [1.0, 2.94726e-07, 4.64528e-09], [1.0, 2.17474e-07, 1.31158e-09], [1.0, 1.7073e-07, 3.47368e-10], [1.0, 1.15248e-07, 1.99951e-10], [1.0, 6.32187e-08, 1.37704e-10], [1.0, 5.11214e-08, 6.7253e-11], [1.0, 3.41383e-08, 6.87201e-11], [1.0, 4.31149e-08, 9.86271e-11], [1.0, 2.76304e-08, 6.4328e-11], [1.0, 2.06728e-08, 3.42118e-11], [1.0, 2.5367e-08, 3.44163e-11], [1.0, 4.68475e-08, 3.69288e-11], [1.0, 1.13977e-07, 5.83321e-11], [1.0, 4.93039e-07, 2.8993e-10], [1.0, 1.52013e-07, 4.66201e-11], [0.999998, 1.921e-06, 1.41816e-09], [0.999997, 2.8869e-06, 2.10946e-09], [0.999995, 5.15103e-06, 2.6403e-09], [0.999999, 1.4423e-06, 7.72265e-10], [0.999997, 3.1645e-06, 1.33235e-09], [0.999996, 4.23939e-06, 1.76276e-09], [0.999993, 6.6659e-06, 1.80138e-09], [0.999994, 6.42836e-06, 3.24499e-09], [0.99999, 1.01223e-05, 2.41499e-09], [0.999993, 7.29493e-06, 1.65171e-09], [0.999998, 1.8575e-06, 4.36861e-10], [0.999997, 2.62075e-06, 8.35763e-10], [0.999998, 2.41401e-06, 5.23877e-10], [0.999999, 1.33471e-06, 1.99645e-10], [0.999998, 2.103e-06, 4.42706e-10], [0.999992, 8.51386e-06, 8.80846e-10], [0.999997, 2.8839e-06, 3.2819e-10], [0.999966, 3.35556e-05, 2.69919e-09], [0.999997, 3.41703e-06, 3.36651e-10], [0.999952, 4.77328e-05, 1.93527e-09], [0.999996, 4.36205e-06, 1.69312e-10], [0.999968, 3.18319e-05, 2.0331e-09], [0.999968, 3.2049e-05, 2.49756e-09], [0.999993, 7.02986e-06, 2.42572e-10], [0.999983, 1.68612e-05, 9.20867e-10], [0.999996, 3.99248e-06, 1.57072e-10], [0.99999, 9.75855e-06, 3.34783e-10], [0.999978, 2.21176e-05, 2.2591e-09], [0.999999, 1.41249e-06, 6.69595e-11], [0.999996, 3.67438e-06, 1.29581e-10], [0.999997, 2.5327e-06, 2.2541e-10], [0.999985, 1.55018e-05, 2.70158e-09], [0.999975, 2.49317e-05, 1.05074e-08], [0.999997, 2.66407e-06, 3.24457e-10], [0.998449, 0.00155046, 1.1678e-07], [0.999322, 0.000678075, 2.16399e-08], [0.923788, 0.0761183, 2.43959e-05], [0.844312, 0.155663, 4.34761e-06], [0.999597, 0.000372579, 7.44037e-06], [0.999915, 4.06417e-05, 1.96509e-05], [0.999748, 0.000208393, 1.29159e-05], [0.999849, 9.17616e-05, 2.69434e-05], [0.999544, 0.000242719, 0.000171737], [0.999663, 0.000152954, 0.000141255], [0.998992, 0.000768927, 0.000180314], [0.999021, 0.000654262, 0.000184598], [0.998873, 0.000924401, 0.000118855], [0.995553, 0.00422427, 0.00010506], [0.99568, 0.00408589, 6.84572e-05], [0.99539, 0.00340683, 0.000718301], [0.983973, 0.0152292, 0.000303596], [0.991176, 0.00743459, 0.000727428], [0.997662, 0.00119601, 0.000451099], [0.997663, 0.00140857, 0.000317425], [0.999362, 0.000265744, 0.000162622], [0.999523, 0.000180849, 0.000161373], [0.999751, 0.000137685, 3.37324e-05], [0.999915, 1.78424e-05, 1.66875e-05], [0.999823, 3.69971e-05, 3.6412e-05], [0.999834, 3.76662e-05, 2.31601e-05], [0.99951, 0.000236243, 6.77033e-05], [0.998743, 0.000942064, 5.21979e-05], [0.998181, 0.00168272, 3.68764e-05], [0.998093, 0.00180493, 3.44701e-05], [0.998172, 0.00173017, 2.43035e-05], [0.998208, 0.00176788, 7.48506e-06], [0.941983, 0.0577381, 0.000197781], [0.862647, 0.136945, 0.000204081], [0.548529, 0.451129, 9.04403e-05], [0.999453, 0.000546181, 2.0949e-07], [0.999199, 0.000799749, 2.24502e-07], [0.999984, 1.63046e-05, 5.4075e-09], [0.999995, 5.05086e-06, 4.87363e-09], [0.999997, 3.28276e-06, 2.47366e-09], [0.999998, 1.8254e-06, 1.37634e-09], [0.999999, 1.29121e-06, 1.40308e-09], [1.0, 2.07057e-07, 4.51157e-10], [1.0, 1.74922e-07, 4.50332e-10], [1.0, 2.42805e-07, 1.19183e-09], [1.0, 6.07218e-08, 3.78633e-10], [1.0, 3.90553e-08, 2.24511e-10], [1.0, 1.01897e-07, 8.04239e-10], [1.0, 6.79215e-08, 4.26298e-10], [1.0, 3.11358e-08, 2.25129e-10], [1.0, 5.86378e-08, 3.49337e-10], [1.0, 4.54619e-08, 2.94319e-10], [1.0, 4.98407e-08, 3.34141e-10], [1.0, 2.15905e-08, 2.33704e-10], [1.0, 3.51525e-08, 2.11634e-10], [1.0, 1.28362e-08, 1.83104e-10], [1.0, 2.61608e-08, 1.72141e-10], [1.0, 7.54167e-09, 1.44369e-10], [1.0, 1.6163e-08, 1.61116e-10], [1.0, 5.68493e-09, 1.473e-10], [1.0, 9.20591e-09, 1.33683e-10], [1.0, 4.784e-09, 1.9562e-10], [1.0, 5.67016e-09, 1.15951e-10], [1.0, 5.92975e-09, 1.04465e-10], [1.0, 3.86814e-09, 1.19722e-10], [1.0, 3.39208e-09, 6.9666e-11], [1.0, 2.3626e-09, 6.46132e-11], [1.0, 1.73602e-09, 9.71885e-11], [1.0, 2.60578e-09, 6.40494e-11], [1.0, 2.78676e-09, 5.6621e-11], [1.0, 1.54349e-09, 7.64852e-11], [1.0, 1.8352e-09, 5.62431e-11], [1.0, 1.48596e-09, 9.51797e-11], [1.0, 1.42545e-09, 5.47979e-11], [1.0, 1.83492e-09, 7.97375e-11], [1.0, 1.68972e-09, 7.63573e-11], [1.0, 1.76545e-09, 8.00081e-11], [1.0, 2.31199e-09, 1.15057e-10], [1.0, 2.85157e-09, 1.31308e-10], [1.0, 2.64652e-09, 1.24005e-10], [1.0, 2.9419e-09, 2.04669e-10], [1.0, 2.44238e-09, 1.68375e-10], [1.0, 1.99456e-09, 1.6174e-10], [1.0, 1.82392e-09, 1.89563e-10], [1.0, 2.18881e-09, 1.93848e-10], [1.0, 1.90131e-09, 2.62578e-10], [1.0, 2.87017e-09, 3.42598e-10], [1.0, 2.84219e-09, 3.4064e-10], [1.0, 3.01309e-09, 3.90647e-10], [1.0, 2.40776e-09, 3.69544e-10], [1.0, 1.95812e-09, 2.79972e-10], [1.0, 1.84476e-09, 2.40509e-10], [1.0, 2.17943e-09, 2.85926e-10], [1.0, 1.82389e-09, 2.85082e-10], [1.0, 2.51281e-09, 2.69969e-10], [1.0, 2.39028e-09, 3.04443e-10], [1.0, 1.71012e-09, 2.27915e-10], [1.0, 2.23822e-09, 3.38449e-10], [1.0, 1.45687e-09, 2.13012e-10], [1.0, 1.93965e-09, 2.50467e-10], [1.0, 2.25466e-09, 3.09512e-10], [1.0, 1.87987e-09, 4.17587e-10], [1.0, 3.0211e-09, 4.88762e-10], [1.0, 2.01296e-09, 4.9583e-10], [1.0, 2.90346e-09, 7.38896e-10], [1.0, 2.68735e-09, 5.64158e-10], [1.0, 2.00012e-09, 4.07657e-10], [1.0, 2.44318e-09, 4.36861e-10], [1.0, 2.54822e-09, 3.36373e-10], [1.0, 3.34394e-09, 6.31386e-10], [1.0, 2.76317e-09, 4.25721e-10], [1.0, 2.38636e-09, 3.17744e-10], [1.0, 2.09303e-09, 3.86006e-10], [1.0, 1.89987e-09, 3.61985e-10], [1.0, 1.80979e-09, 3.74949e-10], [1.0, 2.35668e-09, 3.19356e-10], [1.0, 1.10232e-09, 1.88773e-10], [1.0, 5.8672e-10, 1.70766e-10], [1.0, 3.44153e-10, 1.90228e-10], [1.0, 3.89935e-10, 2.08084e-10], [1.0, 4.39862e-10, 2.74928e-10], [1.0, 6.00843e-10, 2.42285e-10], [1.0, 6.47468e-10, 3.17503e-10], [1.0, 1.00229e-09, 7.79592e-10], [1.0, 9.09349e-10, 3.27079e-10], [1.0, 1.39169e-09, 6.33596e-10], [1.0, 1.91495e-09, 5.53056e-10], [1.0, 1.84962e-09, 6.03457e-10], [1.0, 2.10426e-09, 1.1012e-09], [1.0, 2.11593e-09, 1.53295e-09], [1.0, 3.27096e-09, 2.93432e-09], [1.0, 4.06112e-09, 3.6725e-09], [1.0, 4.18117e-09, 2.7805e-09], [1.0, 4.34405e-09, 3.42769e-09], [1.0, 3.30624e-09, 3.09504e-09], [1.0, 3.70469e-09, 2.41585e-09], [1.0, 3.24373e-09, 3.22961e-09], [1.0, 3.77497e-09, 3.23403e-09], [1.0, 3.72509e-09, 2.91403e-09], [1.0, 3.55129e-09, 2.99725e-09], [1.0, 2.55989e-09, 1.48139e-09], [1.0, 2.2614e-09, 1.66198e-09], [1.0, 2.84029e-09, 1.53128e-09], [1.0, 2.8758e-09, 1.25941e-09], [1.0, 1.8478e-09, 1.74448e-09], [1.0, 1.93715e-09, 1.52153e-09], [1.0, 2.00477e-09, 1.98992e-09], [1.0, 2.39021e-09, 2.27282e-09], [1.0, 2.29692e-09, 1.51265e-09], [1.0, 5.14214e-09, 1.65289e-09], [1.0, 2.3736e-09, 1.59668e-09], [1.0, 5.21892e-09, 3.19072e-09], [1.0, 5.99039e-09, 3.34109e-09], [1.0, 3.6329e-09, 3.05806e-09], [1.0, 2.67783e-09, 2.2598e-09], [1.0, 2.62259e-09, 1.39134e-09], [1.0, 3.9281e-09, 8.58128e-10], [1.0, 2.95239e-09, 7.04538e-10], [1.0, 5.91089e-09, 7.83011e-10], [1.0, 3.08834e-09, 5.28965e-10], [1.0, 3.00075e-09, 9.19038e-10], [1.0, 8.16282e-09, 1.1354e-09], [1.0, 9.90775e-09, 2.03966e-09], [1.0, 2.25909e-08, 5.54572e-09], [1.0, 1.62975e-08, 1.59686e-08], [1.0, 6.17116e-08, 2.06843e-08], [1.0, 2.43841e-08, 1.58642e-08], [1.0, 2.43379e-08, 2.40011e-08], [1.0, 5.56575e-08, 2.36688e-08], [1.0, 1.04469e-07, 2.53814e-08], [1.0, 1.00397e-07, 6.94417e-08], [1.0, 1.48101e-07, 6.28656e-08], [1.0, 2.14927e-08, 1.33349e-08], [1.0, 2.93188e-08, 1.13103e-08], [1.0, 6.15498e-09, 5.83857e-09], [1.0, 5.90187e-09, 3.06441e-09], [1.0, 4.46396e-09, 3.83578e-09], [1.0, 5.87273e-09, 3.7464e-09], [1.0, 7.42383e-09, 6.43448e-09], [1.0, 7.80198e-09, 5.36149e-09], [1.0, 8.84244e-09, 6.5122e-09], [1.0, 9.00397e-09, 7.54216e-09], [1.0, 8.28806e-09, 7.06102e-09], [1.0, 9.26814e-09, 8.43039e-09], [1.0, 8.0464e-09, 7.97954e-09], [1.0, 8.41286e-09, 7.3904e-09], [1.0, 9.05902e-09, 7.08044e-09], [1.0, 1.09339e-08, 7.64637e-09], [1.0, 8.33968e-09, 6.57593e-09], [1.0, 6.6067e-09, 5.56602e-09], [1.0, 8.18763e-09, 5.79473e-09], [1.0, 3.49091e-09, 2.55796e-09], [1.0, 3.77043e-09, 2.76387e-09], [1.0, 3.24784e-09, 2.82895e-09], [1.0, 3.50864e-09, 3.03213e-09], [1.0, 3.25645e-09, 3.19114e-09], [1.0, 4.80079e-09, 4.25382e-09], [1.0, 5.58353e-09, 4.65863e-09], [1.0, 6.44794e-09, 6.33396e-09], [1.0, 7.77198e-09, 6.28704e-09], [1.0, 9.39098e-09, 8.63561e-09], [1.0, 8.63149e-09, 8.48501e-09], [1.0, 1.3329e-08, 1.22799e-08], [1.0, 1.30419e-08, 1.28128e-08], [1.0, 2.09869e-08, 1.97491e-08], [1.0, 2.0544e-08, 1.85548e-08], [1.0, 3.26315e-08, 2.87281e-08], [1.0, 3.01591e-08, 2.45945e-08], [1.0, 4.80144e-08, 4.7961e-08], [1.0, 4.67304e-08, 4.40046e-08], [1.0, 6.75728e-08, 6.15359e-08], [1.0, 7.35907e-08, 6.80467e-08], [1.0, 8.34886e-08, 8.19372e-08], [1.0, 1.4925e-07, 1.47004e-07], [1.0, 1.23642e-07, 1.1551e-07], [0.999999, 2.90149e-07, 2.41883e-07], [0.999999, 2.8323e-07, 2.00862e-07], [1.0, 2.17947e-07, 1.95774e-07], [1.0, 1.99539e-07, 1.38797e-07], [1.0, 1.4973e-07, 1.01262e-07], [1.0, 9.75688e-08, 5.89429e-08], [1.0, 8.24881e-08, 3.54208e-08], [1.0, 6.50886e-08, 3.18944e-08], [1.0, 7.69151e-08, 1.96298e-08], [1.0, 5.37615e-08, 1.01317e-08], [1.0, 7.33376e-08, 4.94409e-08], [1.0, 5.18278e-08, 1.40769e-08], [1.0, 4.14599e-08, 6.56021e-09], [1.0, 5.99721e-08, 3.52777e-08], [1.0, 3.06096e-08, 7.71649e-09], [1.0, 5.43494e-08, 5.05368e-08], [1.0, 6.51961e-08, 4.26876e-08], [1.0, 4.18435e-08, 1.74141e-08], [1.0, 5.41362e-08, 3.57944e-08], [1.0, 7.06377e-08, 1.39953e-08], [1.0, 1.10024e-07, 3.02338e-08], [1.0, 1.1404e-07, 1.26407e-08], [1.0, 2.03921e-07, 4.46047e-08], [1.0, 8.52089e-08, 1.35899e-08], [0.999999, 4.12246e-07, 1.06877e-07], [1.0, 2.20419e-07, 2.34723e-08], [0.999998, 1.59063e-06, 3.88078e-07], [0.999998, 1.69446e-06, 1.34897e-07], [0.999983, 1.49698e-05, 1.52593e-06], [0.99998, 1.96834e-05, 4.87552e-07], [0.999919, 7.89846e-05, 1.65096e-06], [0.999905, 9.38358e-05, 1.05699e-06], [0.999886, 0.000113397, 6.38178e-07], [0.999844, 0.000154206, 9.59418e-07], [0.999821, 0.000177815, 5.30916e-07], [0.999909, 8.99979e-05, 3.71111e-07], [0.999884, 0.00011489, 5.24755e-07], [0.999747, 0.000251888, 4.60681e-07], [0.999743, 0.000255483, 7.60001e-07], [0.999426, 0.000572534, 6.59449e-07], [0.999292, 0.000704433, 1.1238e-06], [0.999785, 0.000212656, 1.03157e-06], [0.999205, 0.000786819, 5.21502e-06], [0.995334, 0.00465299, 2.84356e-06], [0.965736, 0.0341088, 3.63117e-05], [0.941421, 0.0584231, 8.45916e-05], [0.744847, 0.253943, 0.000581366], [0.994618, 0.00440072, 0.000315239], [0.998859, 0.000185489, 0.000175423], [0.999209, 0.000108562, 9.94191e-05], [0.999401, 5.56743e-05, 2.98285e-05], [0.999896, 8.7429e-06, 7.13541e-06], [0.999979, 1.0143e-05, 5.1046e-06]],
#             "top_classes": [[1, 1185, 98], [1, 1185, 783], [1185, 1, 896], [1185, 1, 1118], [1185, 1, 982], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1005], [1185, 1, 1005], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1005, 1], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1005, 1], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1005], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 117], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1118], [1185, 1, 1436], [1185, 1, 1118], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1118], [1185, 1, 212], [1185, 1, 1118], [1185, 1, 212], [1185, 1, 1118], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1910], [1185, 1, 1436], [1185, 1, 1436], [1185, 1, 1910], [1185, 1, 67], [1185, 1, 285], [1185, 1, 67], [1185, 1, 285], [1185, 1, 285], [1185, 1, 92], [1185, 1, 285], [1185, 1, 92], [1, 1185, 1130], [1, 1185, 92], [1, 1185, 1130], [1, 1185, 117], [1, 1185, 92], [1, 117, 92], [1, 1185, 92], [1, 117, 92], [1, 117, 92], [1, 92, 117], [1, 117, 92], [1, 117, 92], [1, 117, 92], [1, 117, 1185], [1, 117, 901], [1, 117, 901], [1, 901, 117], [1, 901, 772], [1, 901, 67], [1, 901, 67], [1, 67, 901], [1, 67, 772], [1, 67, 2528], [1, 1185, 38], [1, 1185, 67], [1, 1185, 67], [1, 117, 1185], [1, 117, 92], [1, 117, 1185], [1, 117, 92], [1, 117, 92], [1, 117, 92], [117, 1, 92], [117, 1, 92], [117, 1, 92], [117, 1, 92], [117, 1, 92], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 1858], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 249], [117, 1, 249], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 249], [117, 1, 2993], [117, 1, 249], [117, 1, 249], [117, 1, 249], [117, 1, 249], [117, 1, 249], [117, 1, 1858], [117, 1, 1858], [117, 1, 249], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 2993], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 2993], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 2993], [117, 1, 1379], [117, 1, 2993], [117, 1, 1858], [117, 1, 1858], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1379, 1], [117, 1, 1858], [117, 1, 1858], [117, 1858, 1], [117, 1858, 1], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1858, 1], [117, 1858, 1], [117, 1858, 1162], [117, 1162, 1858], [117, 1858, 1162], [117, 1, 1162], [117, 1858, 1162], [117, 1, 1858], [117, 1, 1379], [117, 1858, 1162], [117, 1858, 1162], [117, 1, 1379], [117, 1858, 1162], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1, 1858], [117, 1858, 1], [117, 1, 1858], [117, 1858, 1162], [117, 1858, 1], [117, 1858, 1162], [117, 1858, 1162], [117, 1858, 1162], [117, 1162, 1858], [117, 1162, 1858], [117, 1162, 1858], [117, 1, 1379], [117, 1162, 1], [117, 1, 1162], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1], [117, 1379, 1858], [117, 1858, 1], [117, 1, 1858], [117, 1379, 1], [117, 1858, 1], [117, 1, 1379], [117, 1379, 1], [117, 1, 1858], [117, 1858, 1], [117, 1858, 1], [117, 1, 1858], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1858], [117, 1, 1858], [117, 1, 1379], [117, 1, 1858], [117, 1, 1858], [117, 1858, 1], [117, 1858, 1], [117, 1, 1858], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1858], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 1379], [117, 1, 6744], [117, 1, 1379], [117, 1, 6744], [117, 1, 6744], [117, 1, 6744], [117, 1, 6744], [117, 1, 2993], [117, 1, 6744], [1, 117, 6744], [1, 117, 5686], [1, 6744, 117], [1, 5686, 232], [1, 6744, 1051], [1, 6744, 731], [1, 273, 295]]
#         }
#     ]
# }
