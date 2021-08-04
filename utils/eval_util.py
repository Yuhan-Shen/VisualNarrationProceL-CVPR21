import numpy as np
import scipy.optimize


def framewise_eval(pred_list, label_list):
    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    k_pred = int(preds.max()) + 1
    k_label = int(labels.max()) + 1

    overlap = np.zeros([k_pred, k_label])
    for i in range(k_pred):
        for j in range(k_label):
            overlap[i, j] = np.sum((preds==i) * (labels==j))
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-overlap / preds.shape[0])
    K = max(k_pred, k_label)
    
    bg_row_ind = np.concatenate([row_ind, -np.ones(K+1-row_ind.shape[0], dtype=np.int32)])
    bg_col_ind = np.concatenate([col_ind, -np.ones(K+1-col_ind.shape[0], dtype=np.int32)])
    acc = np.mean(bg_col_ind[preds]==bg_row_ind[labels])
    acc_steps = np.mean(bg_col_ind[preds[labels>=0]]==bg_row_ind[labels[labels>=0]])
    
    results = []
    for i, p in enumerate(row_ind):
        correct = preds[labels==col_ind[i]] == p
        if correct.shape[0] == 0:
            num_correct = 0
        else:
            num_correct = np.sum(correct)
        num_label = np.sum(labels==col_ind[i])
        num_pred = np.sum(preds==p)
        results.append([num_correct, num_label, num_pred])

    for i in range(k_pred):
        if i not in row_ind:
            num_correct = 0
            num_label = 0
            num_pred = np.sum(preds==i)
            results.append([num_correct, num_label, num_pred])

    for j in range(k_label):
        if j not in col_ind:
            num_correct = 0
            num_label = np.sum(labels==j)
            num_pred = 0
            results.append([num_correct, num_label, num_pred])

    results = np.array(results)

    precision = np.sum(results[:, 0]) / (np.sum(results[:, 2]) + 1e-10)
    recall = np.sum(results[:, 0]) / (np.sum(results[:, 1]) + 1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)

    return [precision, recall, fscore, acc, acc_steps]

if __name__ == '__main__':
    label_list = [np.random.randint(-1, 5, [i]) for i in np.random.randint(10, 15, 10)]
    lens = [label.shape[0] for label in label_list]
    pred_list = [np.random.randint(-1, 4, [i]) for i in lens]
    metric = framewise_eval(pred_list, label_list)
    print(metric)
            
