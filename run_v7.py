import os
import training_v7

def compute_file_name(pcov, pfc):
    name = ''
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(pfc[0] * 10))
    name += 'fc' + str(int(pfc[1] * 10))
    return name

acc_list = []
count = 0
pcov = [0., 80.]
pfc = [99, 0.]

retrain = 0
lr = 1e-5
f_name = compute_file_name(pcov,pfc)
# pfc[1] = pfc[1] + 5.
# pcov[0] = pcov[0] + 5.
# pcov[1] = pcov[1] + 10.
pfc[0] = 99.1
# pcov[1] = 10.
while (count < 10):
    # pfc[0] = pfc[0] + 0.1
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
        if (retrain > 3):
            print("lowest precision")
            acc_list.append('{},{},{}\n'.format(
                pcov + pfc,
                acc,
                iter_cnt
            ))
            pfc[0] = pfc[0] + 0.1
            # pfc[1] = pfc[1] + 5.
            # pcov[0] = pcov[0] + 5.
            # pcov[1] = pcov[1] + 10.
    else:
        acc_list.append('{},{},{}\n'.format(
            pcov + pfc,
            acc,
            iter_cnt
        ))
        pfc[0] = pfc[0] + 0.1
        # pcov[1] = pcov[1] + 10.
        # pfc[1] = pfc[1] + 5.
        # pcov[0] = pcov[0] + 5.
        # pcov[1] = pcov[1] + 10.
        count = count + 1
        if (retrain != 0):
            retrain = 0

with open("hist.txt","w") as f:
    for item in acc_list:
        f.write(item)
print('accuracy summary: {}'.format(acc_list))
