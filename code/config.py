from os.path import join

model_name = 'CDNet'
test_thread_num = 8  # Thread-num used for test
device = '0'  # GPU index
config.batch_size = 10  # Batch-size used for test

checkpoint_path = None  # The path of pre-trained checkpoint
img_base = ''  # RGB-images base
depth_base = ''  # Depth-maps base
save_base = ''  # Base used to save the predictions

test_roots = {'img': img_base,
              'depth': depth_base}