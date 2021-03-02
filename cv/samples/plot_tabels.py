import numpy as np
import matplotlib.pyplot as plt
import csv

# layer 3 has 32 kernels, each kernel:
# inp=11x11=121
# kernel=7x7=49
# out=9x9=81

iw = 6
kw = 3

inp = []
for r in range(iw):
    _row = []
    for c in range(iw):
        _row.append('i'+str(r*iw+c))
    inp.append(_row)

k = []
for r in range(kw):
    _row = []
    for c in range(kw):
        _row.append('k'+str(r*kw+c))
    k.append(_row)

table_vals = []
for r in range(len(inp)-len(k)+1):
    for c in range(len(inp)-len(k)+1):
        out_c = []
        for p in range(r):
            for _ in range(len(inp)):
                out_c.append(' ')
        for kr in range(len(k)):
            for p in range(c):
                out_c.append(' ')
            for kc in range(len(k)):
                out_c.append('i'+str(r*len(inp)+c+kc+kr*len(inp))+'k'+str(kr*len(k)+kc))
            for p in range(len(inp)-len(k)-c):
                out_c.append(' ')
        for p in range(len(inp)-len(k)-r):
            for _ in range(len(inp)):
                out_c.append(' ')
        table_vals.append(out_c)

tb_np = np.array(table_vals)
tb_np_t = tb_np.T
table_vals = tb_np_t.tolist()

out_len = (len(inp) - len(k) + 1)**2
col_labels = ['out%d' % i for i in range(out_len)]
row_labels = ['inp%d' % i for i in range(len(inp)**2)]

file = open('table1.csv', 'w', newline='')

with file:
    write = csv.writer(file)
    write.writerows(table_vals)


if True:
    # Draw table
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    table1 = plt.table(cellText=table_vals,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)


    table_vals2 = table_vals
    for r in range(len(table_vals2)):
        for c in range(len(table_vals2[r])):
            if table_vals2[r][c].find('i5') != -1:
                table_vals2[r][c] = ' '
            if table_vals2[r][c].find('i2') != -1:
                table_vals2[r][c] = ' '
            if table_vals2[r][c].find('i0') != -1:
                table_vals2[r][c] = ' '
            if table_vals2[r][c].find('i12') != -1:
                table_vals2[r][c] = ' '

            if table_vals2[r][c].find('k0') != -1:
                table_vals2[r][c] = ' '
            if table_vals2[r][c].find('k2') != -1:
                table_vals2[r][c] = ' '

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    table2 = plt.table(cellText=table_vals2, rowLabels=row_labels, colLabels=col_labels, loc='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)

    plt.show()
