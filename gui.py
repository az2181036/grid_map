from PIL import Image
import time


def plot_range_resolution(resolution, st, ed):
    for i in range(st, ed):
        print(i)
        im = Image.open('.\map\\'+str(resolution)+'\Jihua_'+ str(i) + '.pgm')
        im.show()
        time.sleep(1)


def plot_spec_pgm(resolution, pgm_name):
    im = Image.open('.\map\\' + str(resolution) + '\Jihua_' + pgm_name + '.pgm')
    im.show()
    time.sleep(1)


if __name__ == '__main__':
    # plot_spec_pgm(2048, '1009_1012_or')
    plot_spec_pgm(4096, '2002')
    # plot_range_resolution(2048, 1012, 1100)