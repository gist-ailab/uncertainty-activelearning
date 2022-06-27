import random

random.seed(20220622)

dep_path = '/home/hinton/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/rotation'
dest_path = '/home/hinton/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/random'

img_paths = []

for i in range(10):
    f = open(dep_path+f'/loss/batch_{i}.txt', 'r')
    img_paths += f.readlines()
    
print(len(img_paths))
random.shuffle(img_paths)

for i in range(10):
    f = open(dest_path+f'/loss/batch_{i}.txt', 'w')
    for j in range(900):
        
        path = img_paths.pop()
        f.write(path)
    f.close()
        