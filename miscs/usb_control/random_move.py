import numpy as np
from roboarm import Arm


def main():
    arm = Arm()
    random_move(arm)

def rand(a, b, size):
    return (b - a) * np.random.random(size) + a

def random_move(arm, steps = 10, timeout = 1.0):
    for i in range(2):

        move_list = [
            arm.wrist.up,
            arm.wrist.down,
            arm.elbow.up,
            arm.elbow.down,
            arm.shoulder.up,
            arm.shoulder.down,
            arm.base.rotate_clock,
            arm.base.rotate_counter]

        move_list_inv = [
            arm.wrist.down,
            arm.wrist.up,
            arm.elbow.down,
            arm.elbow.up,
            arm.shoulder.down,
            arm.shoulder.up,
            arm.base.rotate_clock,
            arm.base.rotate_counter]

        moves = np.random.randint(0, len(move_list)-2, size=steps)
        print(moves)
        timeout = rand(0.5, 1.0, size = steps)



        for j in range(steps):
            if j % 2 == 0:
                moves[j] = i + len(move_list)-2
                timeout[j] = 1.0

        for j in range(len(moves)):
            move_list[moves[j]](timeout = timeout[j])  

        for j in range(len(moves)):
            move_list_inv[moves[j]](timeout = timeout[j])  
    

if __name__ == '__main__':
    main()
