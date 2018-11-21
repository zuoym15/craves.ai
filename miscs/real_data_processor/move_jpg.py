import os
import shutil

class file_mover(object):
    def __init__(self, root, target_dir):
        self.counter = 0
        self.root = root
        self.target_dir = target_dir

    def move_file(self, file_path):
        _, tempfilename = os.path.split(file_path)
        _, extension = os.path.splitext(tempfilename)
        if extension == '.jpg' or extension == '.png': #only read jpg and png file
            newfile_path = os.path.join(self.target_dir, str(self.counter).zfill(8)+extension)
            shutil.copyfile(file_path, newfile_path)
            print('writing: '+str(self.counter).zfill(8)+extension)
            self.counter += 1
 
    def walk(self, root, path):
        full_path = os.path.join(root, path)
        if os.path.isfile(full_path):
            self.move_file(full_path)

        if os.path.isdir(full_path):
            for dir in os.listdir(full_path):
                self.walk(full_path, dir)
    
    def run(self):
        for dir in os.listdir(self.root):
            self.walk(self.root, dir)

if __name__ == '__main__':
    data_dir = 'C:/Users/Yiming/Desktop/arm_miscs/data'
    root = os.path.join(data_dir, 'img')
    target = os.path.join(data_dir, 'img_all')
    mover = file_mover(root, target)
    mover.run()