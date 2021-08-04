from args import get_args
from data_preprocess import data_preprocess
from train import run_train
from segmentation import run_test

def run_main():
    args = get_args()
    if args.task == 'all':
        task_list = sorted(os.listdir(args.data_dir))
    else:
        task_list = [args.task]

    for task in task_list:
        data_preprocess(args, task)

    for task in task_list:
        run_train(args, task)

    for task in task_list:
        run_test(args, task)


if __name__ == '__main__':
    run_main()
