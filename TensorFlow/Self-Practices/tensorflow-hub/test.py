import tensorflow_hub as hub
import sys

module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v1_50/classification/1')

######################################
# Check input and output information #
######################################
print(module.get_signature_names())
print('\n=====> default'); sys.stdout.flush()
print(module.get_input_info_dict(signature='default')); sys.stdout.flush()
print('\n=====> image_feature_vector'); sys.stdout.flush(); 
print(module.get_input_info_dict(signature='image_feature_vector')); sys.stdout.flush()
print('\n=====> image_classification'); sys.stdout.flush()
print(module.get_input_info_dict(signature='image_classification')); sys.stdout.flush()
print('\n===================\n')
print('=====> default'); sys.stdout.flush()
#print(module.get_output_info_dict(signature='default')); sys.stdout.flush()
[print('{0}: {1}'.format(k, v)) for k, v in sorted(module.get_output_info_dict(signature='default').items())]
print('\n=====> image_feature_vector'); sys.stdout.flush()
#print(module.get_output_info_dict(signature='image_feature_vector')); sys.stdout.flush()
[print('{0}: {1}'.format(k, v)) for k, v in sorted(module.get_output_info_dict(signature='image_feature_vector').items())]
print('\n=====> image_classification'); sys.stdout.flush()
#print(module.get_output_info_dict(signature='image_classification')); sys.stdout.flush()
[print('{0}: {1}'.format(k, v)) for k, v in sorted(module.get_output_info_dict(signature='image_classification').items())]

from Dataset import Dataset
module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v1_50/classification/1', trainable=True)
