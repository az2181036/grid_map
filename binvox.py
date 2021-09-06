import numpy
import numpy as np
import struct
from utils import util


class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz', 'zxy'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        write(self, fp)


def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale


def read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.

    Returns the model with accompanying metadata.

    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 1, 2))
        axis_order = 'xzy'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)


def read_as_sparse(f):
    """ From dense representation to sparse (coordinate) representation.
    No coordinate reordering.
    """
    rst = []
    dims, translate, scale = read_header(f)
    mod1, mod2 = dims[1] * dims[2], dims[2]
    raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    values, counts = raw_data[::2], raw_data[1::2]
    end_indices = np.cumsum(counts)
    st_indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

    for idx, val in enumerate(values):
        if val:
            st = st_indices[idx]
            ed = end_indices[idx]
            for i in range(st, ed):
                x, y, z = util.get_coordinate(i, mod1, mod2)
                rst.append([x, y, z])
    return rst

def sparse_to_dense(voxel_data, dims, dtype=np.bool_):
    if voxel_data.ndim!=2 or voxel_data.shape[0]!=3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims]*3
    dims = np.atleast_2d(dims).T
    # truncate to integers
    xyz = voxel_data.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:,valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out


def read_by_specific_first_dim_plane(fp, st, ed, filepath):
    dims, translate, scale = read_header(fp)
    first_dim_size, second_dim_size, third_dim_size = dims[0], dims[1], dims[2]
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    values, counts = np.array(raw_data[::2], dtype=np.float64), np.array(raw_data[1::2], dtype=np.float64)
    end_index = np.cumsum(counts)

    data = []
    cnt_xy = 0
    cur = st
    st_idx = st * second_dim_size * third_dim_size
    ed_idx = ed * second_dim_size * third_dim_size

    for idx, val in enumerate(end_index):
        if val >= st_idx:
            if idx > 0 and end_index[idx - 1] < st_idx:
                data.extend([values[idx]] * int(val - st_idx))
                cnt_xy += val - st_idx
            if cnt_xy + counts[idx] <= second_dim_size * third_dim_size:
                data.extend([values[idx]] * int(counts[idx]))
                cnt_xy += counts[idx]
            else:
                data.extend([values[idx]] * int(third_dim_size*second_dim_size-cnt_xy))
                if val >= ed_idx:
                    break
                data = np.array(data, np.bool_).reshape(second_dim_size, third_dim_size)
                print(np.sum(data))
                write_to_pgm(data, filepath + '_' + str(cur) + '.pgm')

                cur += 1
                st_idx = cur * third_dim_size * second_dim_size
                cnt_xy = 0
                data = []


def read_by_specific_second_dim_plane(fp, st, ed, filepath):
    dims, translate, scale = read_header(fp)
    first_dim_size, second_dim_size, third_dim_size = dims[0], dims[1], dims[2]
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    values, counts = np.array(raw_data[::2], dtype=np.float64), np.array(raw_data[1::2], dtype=np.float64)
    end_index = np.cumsum(counts)

    for cur in range(st, ed):
    # assume the first dim is 'x' and the third dim is 'y'
        data = []
        cntx, cnt_y = 0, 0
        y_tmp = []
        st_idx = cur * second_dim_size

        for idx, val in enumerate(end_index):
            if val >= st_idx:
                if idx > 0 and end_index[idx-1] < st_idx:
                    y_tmp.extend([values[idx]] * int(val - st_idx))
                    cnt_y += val - st_idx
                if cnt_y + counts[idx] <= third_dim_size:
                    y_tmp.extend([values[idx]] * int(counts[idx]))
                    cnt_y += counts[idx]
                else:
                    y_tmp.extend([values[idx]] * int(third_dim_size-cnt_y))
                    data.append(y_tmp)
                    y_tmp = []
                    cntx += 1
                    if cntx >= first_dim_size:
                        break
                    cnt_y = 0
                    st_idx = cur * second_dim_size + cntx * second_dim_size * third_dim_size
        data = np.array(data, dtype=np.bool_)
        write_to_pgm(data, filepath+'_'+str(cur)+'.pgm')


def write(voxel_model, fp):
    """ Write binary binvox format.

    Note that when saving a model in sparse (coordinate) format, it is first
    converted to dense format.

    Doesn't check if the model is 'sane'.

    """
    if voxel_model.data.ndim==2:
        # TODO avoid conversion to dense
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
    else:
        dense_voxel_data = voxel_model.data

    fp.write('#binvox 1\n')
    fp.write('dim '+' '.join(map(str, voxel_model.dims))+'\n')
    fp.write('translate '+' '.join(map(str, voxel_model.translate))+'\n')
    fp.write('scale '+str(voxel_model.scale)+'\n')
    fp.write('data\n')
    if not voxel_model.axis_order in ('xzy', 'xyz'):
        raise ValueError('Unsupported voxel model axis order')

    if voxel_model.axis_order=='xzy':
        voxels_flat = dense_voxel_data.flatten()
    elif voxel_model.axis_order=='xyz':
        voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c==state:
            ctr += 1
            # if ctr hits max, dump
            if ctr == 255:
                fp.write(chr(state))
                fp.write(chr(ctr))
                ctr = 0
        else:
            # if switch state, dump
            fp.write(chr(state))
            fp.write(chr(ctr))
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        fp.write(chr(state))
        fp.write(chr(ctr))


def get_pgm_from_voxel(voxel, filepath):
    size = voxel.dims[-1]
    data = voxel.data
    # data = voxel.data[504] + voxel.data[506]
    for i in range(size):
        # print(np.sum(data[i]))
        write_to_pgm(data[i], filepath+'_'+str(i)+'.pgm')


def get_two_pgm_or(voxel, filepath, or_idx):
    assert isinstance(or_idx, list)
    if len(or_idx) == 0:
        print("or_idx must have more than 1 element.")
        return

    data = voxel.data[or_idx[0]]
    for idx in or_idx[1:]:
        data = data + voxel.data[idx]
    write_to_pgm(data, filepath+'_'+'_'.join(map(lambda x:str(x), or_idx))+'_or.pgm')


def write_to_pgm(xy_data, filepath):
    pgm_size = xy_data.shape
    with open(filepath, 'wb') as fout:
        if len(xy_data) == 0:
            header = 'P5\n0\n255\n'
        else:
            header = 'P5\n'+str(pgm_size[0]) + ' ' + str(pgm_size[1]) + '\n255\n'
        fout.write(header.encode())
        for i in range(pgm_size[0]):
            for j in range(pgm_size[1]):
                if xy_data[i][j]:
                    fout.write(struct.pack('B', 0))
                else:
                    fout.write(struct.pack('B', 255))


def draw_image(xy_data, filepath):
    from PIL import Image
    size = xy_data.shape
    gray = []
    for i in range(size[0]):
        tmp = []
        for j in range(size[1]):
            if xy_data[i][j]:
                tmp.append(255)
            else:
                tmp.append(0)
        gray.append(tmp)
    gray = np.array(gray, dtype=np.uint8)
    print(gray.shape)
    im = Image.fromarray(gray)
    im.show()


if __name__ == '__main__':
    x1, x2 = [], []
    filewritepath = ".\map\\4096\Jihua"
    with open('.\map\\NewWorld.ply_4096.binvox', 'rb') as f:
        #m1 = read_as_3d_array(f, True)
        # get_pgm_from_voxel(m1, filewritepath)
        # get_two_pgm_or(m1, filewritepath, [1009, 1012])

        read_by_specific_second_dim_plane(f, 0, 4096, filewritepath)

    # draw_image(m1.data[4],'.\map\gray.bmp')
