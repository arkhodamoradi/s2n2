import numpy as np
import matplotlib.pyplot as plt
import time
from heapq import heappush, heappop, heapify
from collections import defaultdict, Counter


def get_errors():
    ps = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    images = 10
    im_mul = 0
    ts = 500

    errors = np.zeros([len(ps), images, 32, ts])

    w0 = np.load('iwb500/w0.npy')  # 32 x 24 x 7 x 7
    b0 = np.load('iwb500/b0.npy')  # 32

    pp = -1
    for p in ps:
        pp = pp + 1
        print('processing: {}'.format(pp))
        for t in range(ts):
            f_name = 'iwb500/i' + str(t) + '.npy'
            inp = np.load(f_name)  # 512 x 24 x 11 x 11
            for i in range(images):  # 512
                for k in range(32):
                    kernel = w0[k]
                    bias = b0[k]

                    idx_p = np.where(kernel > 0)
                    idx_n = np.where(kernel < 0)

                    kernel_p = kernel[idx_p]
                    k_idx_p = np.flip(np.argsort(kernel_p), 0)
                    _lp = int(k_idx_p.size * p)
                    k_idx_p = k_idx_p[0:_lp]

                    kernel_n = kernel[idx_n]
                    k_idx_n = np.argsort(kernel_n)
                    _ln = int(k_idx_n.size * p)
                    k_idx_n = k_idx_n[0:_ln]

                    diff = 0
                    diff_s = 0

                    for w in range(4):
                        for h in range(4):
                            data = inp[i + (im_mul * images), :, 0 + w:7 + w, 0 + h:7 + h]
                            mul = np.multiply(data, kernel)
                            out_c = np.sum(mul) + bias
                            out_complete = 1 if out_c > 0 else 0

                            data_p = data[idx_p]
                            data_n = data[idx_n]

                            mul_p = np.multiply(data_p[k_idx_p], kernel_p[k_idx_p])
                            mul_n = np.multiply(data_n[k_idx_n], kernel_n[k_idx_n])
                            out_d = np.sum(mul_p) + np.sum(mul_n) + bias
                            out_deducted = 1 if out_d > 0 else 0
                            _d = 1 if out_complete != out_deducted else 0
                            diff = diff + _d
                            diff_s = diff_s + 1

                    error = diff / diff_s
                    errors[pp, i, k, t] = error

    np.save('iwb500/errors', errors)


def plot_errors(kernel):
    errors = np.load('iwb/errors.npy')
    mean_errors_images = np.mean(errors, 1)  # this is ps, kernels, ts
    ps = mean_errors_images[:, kernel, :]

    plt.figure()
    plt.plot(ps[0], 'b')
    plt.plot(ps[1], 'g')

    plt.figure()
    plt.plot(ps[2], 'b')
    plt.plot(ps[3], 'g')

    plt.figure()
    plt.plot(ps[4], 'b')
    plt.plot(ps[5], 'g')

    plt.figure()
    plt.plot(ps[6], 'b')
    plt.plot(ps[7], 'g')

    plt.figure()
    plt.plot(ps[8], 'b')
    plt.plot(ps[9], 'g')
    plt.show()


def learn_ps(k, images, im_mul, error_limit):

    w0 = np.load('iwb500/w0.npy')  # 32 x 24 x 7 x 7
    b0 = np.load('iwb500/b0.npy')  # 32

    ts = 500

    kernel = w0[k]
    bias = b0[k]
    idx_p = np.where(kernel > 0)
    idx_n = np.where(kernel < 0)

    kernel_p = kernel[idx_p]
    k_idx_p_fixed = np.flip(np.argsort(kernel_p), 0)

    kernel_n = kernel[idx_n]
    k_idx_n_fixed = np.argsort(kernel_n)

    ps = []
    for t in range(ts):
        print('processing {}'.format(t))
        error = 1
        p = 0.0
        p_step = 0.05
        f_name = 'iwb500/i' + str(t) + '.npy'
        inp = np.load(f_name)  # 512 x 24 x 11 x 11
        while error > error_limit:
            p = p + p_step

            _lp = int(k_idx_p_fixed.size * p)
            k_idx_p = k_idx_p_fixed[0:_lp]

            _ln = int(k_idx_n_fixed.size * p)
            k_idx_n = k_idx_n_fixed[0:_ln]

            errors = np.zeros([images, ])
            for i in range(images):  # 512
                diff = 0
                diff_s = 0
                for w in range(4):
                    for h in range(4):
                        data = inp[i + (im_mul * images), :, 0 + w:7 + w, 0 + h:7 + h]
                        mul = np.multiply(data, kernel)
                        out_c = np.sum(mul) + bias
                        out_complete = 1 if out_c > 0 else 0

                        data_p = data[idx_p]
                        data_n = data[idx_n]

                        mul_p = np.multiply(data_p[k_idx_p], kernel_p[k_idx_p])
                        mul_n = np.multiply(data_n[k_idx_n], kernel_n[k_idx_n])
                        out_d = np.sum(mul_p) + np.sum(mul_n) + bias
                        out_deducted = 1 if out_d > 0 else 0
                        _d = 1 if out_complete != out_deducted else 0
                        diff = diff + _d
                        diff_s = diff_s + 1

                errors[i] = diff / diff_s

            error = np.mean(errors, 0)

        print('\t p: {}'.format(p))
        ps.append(p)

    return ps


def scan_ranges():
    # we have 100_{1, 2, 3, 4}
    limits = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    im_ss = [100, 200, 500]
    for im_s in im_ss:
        _s = time.time()
        print('processing im: {}'.format(im_s))
        for lim in limits:
            print('processing lim: {}'.format(lim))
            ps_k = []
            for ker in range(32):
                print('processing kernel: {}'.format(ker))
                ps_i = learn_ps(k=ker, images=im_s, im_mul=0, error_limit=lim)
                ps_k.append(ps_i)
            f_name_ = 'ps/p_' + str(im_s) + '_' + str(int(lim*100))
            np.save(f_name_, ps_k)
        print('processed {} in {} minutes.'.format(im_s, (time.time() - _s)/60))


def plot_ranges():

    from mpl_toolkits.mplot3d import axes3d
    kernels1 = np.load('ps/p_100_1.npy')
    kernels2 = np.load('ps/p_100_2.npy')
    kernels3 = np.load('ps/p_100_8.npy')
    kernels4 = np.load('ps/p_100_9.npy')
    x = np.arange(0, 500, 1)
    y = np.arange(0, 32, 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_zlim(0, 1)
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_zlim(0, 1)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_zlim(0, 1)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_zlim(0, 1)

    mycmap = plt.get_cmap('gist_earth')
    ax1.plot_surface(x, y, kernels1, cmap=mycmap)
    ax2.plot_surface(x, y, kernels2, cmap=mycmap)
    ax3.plot_surface(x, y, kernels3, cmap=mycmap)
    ax4.plot_surface(x, y, kernels4, cmap=mycmap)

    plt.show()


def first_fit_decreasing(kernels, k_size=1176):

    class sub_k:
        def __init__(self, idx, v):
            self.idxs = [idx]
            self.vals = [v]
            self.sum = v

        def add_kernel(self, idx, v):
            self.idxs.append(idx)
            self.vals.append(v)
            self.sum += v

    unsorted_idx = np.flip(np.argsort(kernels), 0)
    sorted_kernels = np.flip(np.sort(kernels), 0)

    subs = []
    for k in range(len(sorted_kernels)):
        placed = False
        for s in range(len(subs)):
            if (int(sorted_kernels[k]*k_size) + subs[s].sum) <= k_size:
                subs[s].add_kernel(unsorted_idx[k], int(sorted_kernels[k]*k_size))
                placed = True
                break
        if not placed:
            sub = sub_k(unsorted_idx[k], int(k_size*sorted_kernels[k]))
            subs.append(sub)

    return subs


def schedule(lim):

    f_name = 'ps/p_100_' + str(lim) + '.npy'
    kernels_ts = np.load(f_name)
    time_steps = kernels_ts.shape[1]
    lens = []
    sch_ts = []
    for ts in range(time_steps):
        kernels = kernels_ts[:, ts]
        subs = first_fit_decreasing(kernels)
        sch_ts.append(subs)
        length = len(subs)
        lens.append(length)
    return lens, sch_ts


def check_inp(ts=0):
    channels_nz = []
    f_name = 'iwb500/i' + str(ts) + '.npy'
    inp = np.load(f_name)  # 512*24*11*11
    inp = np.reshape(inp, (512, 24, 121))
    for i in range(24):
        nz = 0
        for b in range(512):
            nz = nz + np.nonzero(inp[b, i, :])[0].size
        channels_nz.append(nz/512)
    return channels_nz


def plot_nz():
    channels_nz = np.array(check_inp(0))
    for i in range(1, 500):
        channels_nz = channels_nz + np.array(check_inp(i))
    channels_nz = channels_nz / 500
    print(np.where(channels_nz < 60))
    print(np.where(channels_nz < 90))
    plt.plot(channels_nz)
    plt.show()


def plot_nz_img(c=0):
    img = np.zeros((11, 11))
    for i in range(500):
        f_name = 'siwb500/i' + str(i) + '.npy'
        inp = np.load(f_name)[:, c, :, :]
        inp[inp > 0] = 1
        inps = np.sum(inp, 0)
        img = img + np.array(inps)
    img = img / 512
    plt.imshow(img, cmap='gray')
    plt.show()


def naive_comp(inp):
    res = []
    cur = inp[0]
    res.append(cur)
    cntr = 1
    new_cntr = False
    for i in range(1, inp.size):
        if cur == inp[i]:
            cntr = cntr + 1
            new_cntr = True
        else:
            res.append(cntr)
            cntr = 1
            new_cntr = False
            cur = inp[i]
    if new_cntr:
        res.append(cntr)
    return res


def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def get_mask(w0k, p=0.5):
    s = w0k.shape[0]*w0k.shape[1]*w0k.shape[2]
    w00 = np.reshape(w0k, (s,))
    idx = np.flip(np.argsort(w00), 0)
    mask = np.zeros_like(idx)
    mask[idx[0:int(p*s)]] = 1
    return mask


def huffman_compress(w0k):
    mask = w0k#get_mask(w0k)
    compressed = naive_comp(mask)
    print(np.unique(compressed))

    symb2freq = defaultdict(int)
    symb2freq = Counter(compressed)
    huff = encode(symb2freq)
    #print('Symbol\tWeight\tHuffman Code\tTotal bits')
    total = 0
    for p in huff:
    #    print("%s\t%s\t%s\t%s" % (p[0], symb2freq[p[0]], p[1], symb2freq[p[0]]*len(p[1])))
        total += symb2freq[p[0]]*len(p[1])
    print('total bits: {}'.format(total))
    print(compressed)
    print(huff)


def get_unique(w0k=0):
    mask = get_mask(w0k)
    compressed = naive_comp(mask)
    return np.unique(compressed)


def split_array(arr, ptrn):
    proc_arr = []
    last_i = 0
    i = 0
    while i < arr.size:
        if i+ptrn.size <= arr.size:
            if np.array_equal(ptrn, arr[i:i+ptrn.size]):
                proc_arr.append(arr[last_i:i])
                i += ptrn.size
                last_i = i
            else:
                i += 1
                proc_arr.append(arr[last_i:i])
                last_i = i
        else:
            proc_arr.append(arr[last_i:i])
            break

    clean_arr = [elem for elem in proc_arr if elem.size > 0]
    return clean_arr


def get_max_freq(arr, ptrn):
    # start from 0 and shift to the length of pattern, find maximum freq
    freqs = []
    for j in range(ptrn.size):
        freq = 0
        i = 0
        arr_t = arr[j:arr.size]
        while i < arr_t.size:
            if i+ptrn.size <= arr_t.size:
                if np.array_equal(ptrn, arr_t[i:i+ptrn.size]):
                    freq += 1
                    i += ptrn.size
                else:
                    i += 1
            else:
                break
        freqs.append(freq)

    return np.array(freqs), np.max(freqs)


def make_array(arrs):
    mask = arrs[0].copy()
    depth = len(arrs) - 1
    for i in range(depth):
        cntr = 0
        for j in range(mask.size):
            if mask[j] == 0:
                mask[j] = arrs[i+1][cntr]
                cntr += 1
    return mask


def make_array_HLS(arrs, ml=0, depth=0):
    mask = arrs[0:ml].copy()
    skip = 16
    cntr = 0
    for i in range(depth):
        skip = skip + cntr
        cntr = 0
        for j in range(ml):
            if mask[j] == 0:
                mask[j] = arrs[skip+cntr]
                cntr += 1
    return mask


def mask_conv():
    ts = 500
    images = 512

    w0 = np.load('siwb500/w0.npy')  # 32 x 24 x 7 x 7
    b0 = np.load('siwb500/b0.npy')  # 32

    # errors = np.zeros([images, 32, ts])
    dist_avg_spike_p = np.zeros([images, 32, ts])
    dist_avg_nospike_p = np.zeros([images, 32, ts])
    dist_avg_spike_n = np.zeros([images, 32, ts])
    dist_avg_nospike_n = np.zeros([images, 32, ts])

    for t in range(ts):
        f_name = 'siwb500/i' + str(t) + '.npy'
        inp = np.load(f_name)  # 512 x 24 x 11 x 11
        f_name = 'siwb500/s' + str(t) + '.npy'
        inp_s = np.load(f_name)  # 512 x 24 x 11 x 11
        for i in range(images):  # 512
            for k in range(32):
                kernel = w0[k]
                bias = b0[k]

                mask_p = np.zeros_like(kernel)
                idx_p = np.where(kernel > 0)
                mask_p[idx_p] = 1

                mask_n = np.zeros_like(kernel)
                idx_n = np.where(kernel < 0)
                mask_n[idx_n] = 1

                # diff = 0
                # diff_s = 0

                dist_spike_p = 0
                dist_spike_n = 0
                dist_spike_s = 0

                dist_nospike_p = 0
                dist_nospike_n = 0
                dist_nospike_s = 0

                for w in range(4):
                    for h in range(4):
                        data = inp[i, :, w:7+w, h:7+h]
                        mul = np.multiply(data, kernel)
                        out_c = np.sum(mul) + bias
                        out_complete = 1 if out_c > 0 else 0

                        data_s = inp_s[i, :, w:7+w, h:7+h]  # input 0, 1
                        masked_data_p = np.multiply(data_s, mask_p)  # input 1s with positive weight
                        masked_data_n = np.multiply(data_s, mask_n)  # input 1s with negative weight

                        mul_p = np.multiply(masked_data_p, mul)
                        mul_n = np.multiply(masked_data_n, mul)

                        if bias > 0:
                            res_p = np.sum(mul_p) + bias
                            res_n = np.sum(mul_n)
                        else:
                            res_p = np.sum(mul_p)
                            res_n = np.sum(mul_n) + bias

                        # out_deducted = 0
                        # if np.sum(mul_p)+bias > np.sum(mul_n):
                        #    out_deducted = out_complete

                        if out_complete == 1:
                            dist_spike_p += res_p
                            dist_spike_n += res_n
                            dist_spike_s += 1
                        else:
                            dist_nospike_p += res_p
                            dist_nospike_n += res_n
                            dist_nospike_s += 1

                        # _d = 1 if out_complete != out_deducted else 0
                        # diff += _d
                        # diff_s += 1

                # if diff_s != 0:
                #    errors[i, k, t] = diff / diff_s
                # else:
                #    errors[i, k, t] = 0

                if dist_spike_s != 0:
                    dist_avg_spike_p[i, k, t] = dist_spike_p / dist_spike_s
                    dist_avg_spike_n[i, k, t] = dist_spike_n / dist_spike_s
                else:
                    dist_avg_spike_p[i, k, t] = 0
                    dist_avg_spike_n[i, k, t] = 0

                if dist_nospike_s != 0:
                    dist_avg_nospike_p[i, k, t] = dist_nospike_p / dist_nospike_s
                    dist_avg_nospike_n[i, k, t] = dist_nospike_n / dist_nospike_s
                else:
                    dist_avg_nospike_p[i, k, t] = 0
                    dist_avg_nospike_n[i, k, t] = 0

    np.save('siwb500/dist_avg_Spike_abs_p', dist_avg_spike_p)
    np.save('siwb500/dist_avg_Spike_abs_n', dist_avg_spike_n)
    np.save('siwb500/dist_avg_NoSpike_abs_p', dist_avg_nospike_p)
    np.save('siwb500/dist_avg_NoSpike_abs_n', dist_avg_nospike_n)

    return dist_avg_spike_p, dist_avg_spike_n, dist_avg_nospike_p, dist_avg_nospike_n


plot_ranges()

if False:
    #distSpike_p, distSpike_n, distNoSpike_p, distNoSpike_n = mask_conv()

    #distSpike = np.load('siwb500/dist_avg_Spike.npy')
    #distNoSpike = np.load('siwb500/dist_avg_NoSpike.npy')
    distSpike_p = np.load('siwb500/dist_avg_Spike_abs_p.npy')
    distSpike_n = np.load('siwb500/dist_avg_Spike_abs_n.npy')
    distNoSpike_p = np.load('siwb500/dist_avg_NoSpike_abs_p.npy')
    distNoSpike_n = np.load('siwb500/dist_avg_NoSpike_abs_n.npy')

    distSpike_p = np.mean(distSpike_p, 0)
    distSpike_n = np.mean(distSpike_n, 0)
    distNoSpike_p = np.mean(distNoSpike_p, 0)
    distNoSpike_n = np.mean(distNoSpike_n, 0)

    diff_spike = np.load('siwb500/dist_avg_Spike.npy')
    diff_nospike = np.load('siwb500/dist_avg_NoSpike.npy')

    diff_spike = np.mean(diff_spike, 0)
    diff_nospike = np.mean(diff_nospike, 0)

    bias = np.load('siwb500/b0.npy')
    print(bias.shape)

    for i in range(32):
        plt.figure()
        plt.plot(distSpike_p[i], 'b')
        plt.plot(distNoSpike_p[i], 'k')
        plt.plot(distSpike_n[i], 'g')
        plt.plot(distNoSpike_n[i], 'r')

    plt.show()

    #arr_25 = np.array([1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0])
    #arr_50 = np.array([1,0,1,0,0,1,1,0,0,0,0,0])
    #arr_75 = np.array([0,0,1,0,1,1,0,1])
    #print(make_array([arr_25, arr_50, arr_75]))
    #arr_s = np.concatenate((arr_25, arr_50))
    #arr_s = np.concatenate((arr_s, arr_75))
    #print(make_array_HLS(arr_s, ml=16, depth=2))
