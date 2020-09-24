from solver import Solver
import config
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    
    solver = Solver()
    solver.test(ckpt_path=config.checkpoint_path, 
                test_roots=config.test_roots,
                batch_size=config.batch_size,
                test_thread_num=config.test_thread_num,
                save_base=config.save_base)