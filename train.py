import subprocess
import os


task = os.getenv('task', 'fsod')
vit = os.getenv('vit', 'l')
#dataset = os.getenv('dataset', 'coco')
dataset = os.getenv('dataset', 'coco')
shot = os.getenv('shot', '30')
split = os.getenv('split', '1')


def get_num_gpus():
    try:
        result = subprocess.check_output("nvidia-smi -L", shell=True).decode('utf-8')
        gpu_count = result.count('GPU')
        return str(gpu_count) if gpu_count > 0 else '1'
    except subprocess.CalledProcessError as e:
        print(f"Error fetching GPU count: {e}")
        return '1'


num_gpus = 4

print(f"task={task}, vit={vit}, dataset={dataset}, shot={shot}, split={split}, num_gpus={num_gpus}")


def run_command(command):
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)


if task == 'ovd':
    if dataset == 'coco':
        command = (
            f"python /root/PiDiViT/tools/train_net.py --num-gpus {num_gpus} "
            f"--config-file /root/PiDiViT/configs/open-vocabulary/coco/vit{vit}.yaml "
            f"MODEL.WEIGHTS /root/PiDiViT/weights/initial/open-vocabulary/vit{vit}+rpn.pth "
            f"DE.OFFLINE_RPN_CONFIG /root/PiDiViT/configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml "
            f"OUTPUT_DIR /root/PiDiViT/output/train/open-vocabulary/coco/vit{vit}/"
        )

    else:
        command = (
            f"python /root/PiDiViT/tools/train_net.py --num-gpus {num_gpus}"
            f"--config-file /root/PiDiViT/configs/open-vocabulary/lvis/vit{vit}.yaml "
            f"MODEL.WEIGHTS /root/PiDiViT/weights/initial/open-vocabulary/vit{vit}+rpn_lvis.pth "
            f"DE.OFFLINE_RPN_CONFIG /root/PiDiViT/configs/RPN/mask_rcnn_R_50_FPN_1x.yaml "
            f"OUTPUT_DIR /root/PiDiViT/output/train/open-vocabulary/lvis/vit{vit}/"
        )
    run_command(command)

elif task == 'fsod':
    command = (
        f"python /root/PiDiViT/tools/train_net.py --num-gpus {num_gpus} "
        f"--config-file /root/PiDiViT/configs/few-shot/vit{vit}_shot{shot}.yaml "
        f"MODEL.WEIGHTS /root/PiDiViT/weights/initial/few-shot/vit{vit}+rpn.pth "
        f"DE.OFFLINE_RPN_CONFIG /root/PiDiViT/configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml "
        f"OUTPUT_DIR /root/PiDiViT/output/train/few-shot/shot-{shot}/vit{vit}/"
    )
    run_command(command)

elif task == 'osod':
    command = (
        f"python /root/PiDiViT/tools/train_net.py "
        f"--num-gpus {num_gpus} "
        f"--config-file /root/PiDiViT/configs/one-shot/split{split}_vit{vit}.yaml "
        f"MODEL.WEIGHTS /root/PiDiViT/weights/initial/oneshot/vit{vit}+rpn.split{split}.pth "
        f"DE.OFFLINE_RPN_CONFIG /root/PiDiViT/configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml "
        f"OUTPUT_DIR /root/PiDiViT/output/train/one-shot/split{split}/vit{vit}/ "
        f"DE.ONE_SHOT_MODE True"
    )
    run_command(command)

else:
    print("skip")
