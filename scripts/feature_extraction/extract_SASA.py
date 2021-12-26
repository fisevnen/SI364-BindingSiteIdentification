from pymol import cmd, stored
import os

def calcResidueSasa(protein, path, outpath):
    cmd.delete('all')
    cmd.load(path + '/' + protein)

    cmd.remove('h.')
    cmd.remove('solvent')
    cmd.remove('org.')
    cmd.remove('metals')


    cmd.set('dot_solvent', 1) # dot_solvent = 0 means calculating the vdw surface area; dot_solvent = 1 means calculating the SASA
    cmd.set('dot_density', 3) # ranging from 1 to 4(int), 4 is the slowest but the most accurate, 1 is the fastest but the most rough

    stored.residues = []
    cmd.iterate('name CA', 'stored.residues.append(resi)')

    residue_sasa = {}
    for i in stored.residues:
        residue_sasa[i] = cmd.get_area('resi %s' % i)

    with open(outpath + '/' + protein[:-4] +'-sasa.txt','w') as f:
        for j in residue_sasa.keys():
            f.write(j + ' ' + '%.2f' %residue_sasa[j] + '\n')
        f.close()


def fuc(msg,):
    print(msg)
    time.sleep(10)
        
from multiprocessing import Pool
import multiprocessing
import time

if __name__ == "__main__":
    file_list = os.listdir('part2/pdb')

        
    pool1 = Pool(processes=60)

    for i in range(len(file_list)):
        pool1.apply_async(calcResidueSasa, args=(file_list[i], 'part2/pdb', 'part2/sasa'))
        # pool1.apply_async(fuc, args=(i,))

    pool1.close()
    pool1.join()

# import multiprocessing
# import time
# def worker(msg):
#     print ("#######start {0}########".format(msg))
#     time.sleep(1)
#     print ("#######end   {0}########".format(msg))
 
# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=3)
#     for i in range(1, 10):
#         msg = "hello{0}".format(i)
#         pool.apply_async(func=worker, args=(msg,))
#     pool.close()
#     pool.join()     #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
#     print ("main end")