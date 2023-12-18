def _get_val_indx(utt, target):
    idx_list = []
    targ_idx = 0
    for i in range(len(utt)):
        if utt[i] == target[targ_idx]:
            targ_idx += 1
            idx_list.append(i)

    return idx_list


print(_get_val_indx("上海海交通通大", "上海交通大学"))
