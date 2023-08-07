import tensorflow_datasets as tfds
# import medical_ts_datasets

physionet_dataset = tfds.load(name='mimic3_mortality', split='train')
physionet_dataset = tfds.load(name='mimic3_mortality', split='test')
physionet_dataset = tfds.load(name='mimic3_mortality', split='valid')