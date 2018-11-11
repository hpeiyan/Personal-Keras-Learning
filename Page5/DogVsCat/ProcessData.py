import os, shutil
import sys
sys.path.append('../..')
import Utils as ut

INFO = '[INFO] '


class PreProcessData:
    @staticmethod
    def process():
        print(INFO + 'Processing Start.')
        # original_data_dir = r'/Users/Yuseng/Downloads/Deep-Learning-For-Computer-Vision-master/datasets/animals'
        # original_data_dir = r'/Users/Yuseng/Downloads/all/train'
        # original_data_dir = r'/home/bigdata/Documents/DeepLearningProject/CatVsDog/train'
        # original_data_dir = r'/Users/zzc20160628-14/Downloads/cat_dog_data/train'
        original_data_dir = r'/home/ubuntu/DeepLearningProject/data/train'
        base_dir = './cat_and_dog_small'
        ut.ifNoneCreateDirs(base_dir)

        train_dir = os.path.join(base_dir, 'train')
        ut.ifNoneCreateDirs(train_dir)

        val_dir = os.path.join(base_dir, 'validation')
        ut.ifNoneCreateDirs(val_dir)

        test_dir = os.path.join(base_dir, 'test')
        ut.ifNoneCreateDirs(test_dir)

        cat_train_dir = os.path.join(train_dir, 'cat')
        ut.ifNoneCreateDirs(cat_train_dir)

        dog_train_dir = os.path.join(train_dir, 'dog')
        ut.ifNoneCreateDirs(dog_train_dir)

        cat_val_dir = os.path.join(val_dir, 'cat')
        ut.ifNoneCreateDirs(cat_val_dir)

        dog_val_dir = os.path.join(val_dir, 'dog')
        ut.ifNoneCreateDirs(dog_val_dir)

        cat_test_dir = os.path.join(test_dir, 'cat')
        ut.ifNoneCreateDirs(cat_test_dir)

        dog_test_dir = os.path.join(test_dir, 'dog')
        ut.ifNoneCreateDirs(dog_test_dir)

        fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
        for name in fnames:
            src = os.path.join(original_data_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError
            dst = os.path.join(cat_train_dir, name)
            if os.path.exists(dst):
                continue
            shutil.copy(src=src, dst=dst)

        fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
        for name in fnames:
            src = os.path.join(original_data_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError
            dst = os.path.join(cat_val_dir, name)
            if os.path.exists(dst):
                continue
            shutil.copy(src=src, dst=dst)

        fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
        for name in fnames:
            src = os.path.join(original_data_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError
            dst = os.path.join(cat_test_dir, name)
            if os.path.exists(dst):
                continue
            shutil.copy(src=src, dst=dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
        for name in fnames:
            src = os.path.join(original_data_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError
            dst = os.path.join(dog_train_dir, name)
            if os.path.exists(dst):
                continue
            shutil.copy(src=src, dst=dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
        for name in fnames:
            src = os.path.join(original_data_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError
            dst = os.path.join(dog_val_dir, name)
            if os.path.exists(dst):
                continue
            shutil.copy(src=src, dst=dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
        for name in fnames:
            src = os.path.join(original_data_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError
            dst = os.path.join(dog_test_dir, name)
            if os.path.exists(dst):
                continue
            shutil.copy(src=src, dst=dst)

        print(INFO + 'Processing End.')
        return train_dir, val_dir, test_dir
