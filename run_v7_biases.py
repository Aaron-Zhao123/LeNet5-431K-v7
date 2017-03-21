import os
import training_v7_biases

def compute_file_name(pcov, pfc):
    name = 'biases'
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(pfc[0] * 10))
    name += 'fc' + str(int(pfc[1] * 10))
    return name

acc_list = []
count = 0
pcov = [0., 0.]
pfc = [0., 0.]

retrain = 0
lr = 1e-4
f_name = compute_file_name(pcov,pfc)

while (count < 10):
    pfc[0] = pfc[0] + 10.
    # pfc[1] = pfc[1] + 5.
    # pcov[0] = pcov[0] + 10.
    # pcov[1] = pcov[1] + 10.
    if (retrain == 0):
        lr = 1e-4
    # prune
    param = [
    ('-pcov',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-fname',f_name),
    ('-lr',lr),
    ('-PRUNE',True),
    ('-TRAIN',False),
    ('-parent_dir','./')
    ]

    _ = training_v7.main(param)
    f_name = compute_file_name(pcov,pfc)

    # train pruned model
    param = [
    ('-pcov',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-fname',f_name),
    ('-lr',lr),
    ('-PRUNE', False),
    ('-TRAIN', True),
    ('-parent_dir','./')
    ]
    acc,iter_cnt = training_v7.main(param)

    if (acc < 0.9936):
        retrain += 1
        lr = lr / float(2)
        if (retrain > 2 or pfc[1] > 90):
            print("lowest precision")
            # break
            acc_list.append('{},{},{}\n'.format(
                pcov + pfc,
                acc,
                iter_cnt
            ))
    else:
        acc_list.append('{},{},{}\n'.format(
            pcov + pfc,
            acc,
            iter_cnt
        ))
        count = count + 1
        if (retrain != 0):
            retrain = 0

with open("biases_hist.txt","w") as f:
    for item in acc_list:
        f.write(item)
print('accuracy summary: {}'.format(acc_list))
