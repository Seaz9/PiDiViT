import argparse
import subprocess
import pynvml


def main(task, vit, dataset, shot, split, num_gpus):
    print(f"task={task}, vit={vit}, dataset={dataset}, shot={shot}, split={split}, num_gpus={num_gpus}")

    if task == "ovd":
        if dataset == "coco":
            cmd = [
                "python3", "tools/train_net.py",
                "--num-gpus", str(num_gpus), "--eval-only",
                "--config-file", f"configs/open-vocabulary/coco/vit{vit}.yaml",
                "MODEL.WEIGHTS",
                subprocess.check_output(f"ls weights/trained/open-vocabulary/coco/vit{vit}_*.pth | head -n 1",
                                        shell=True).strip(),
                "DE.OFFLINE_RPN_CONFIG", "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml",
                "OUTPUT_DIR", f"output/eval/open-vocabulary/coco/vit{vit}/"
            ]
        else:
            cmd = [
                "python3", "tools/train_net.py",
                "--num-gpus", str(num_gpus), "--eval-only",
                "--config-file", f"configs/open-vocabulary/lvis/vit{vit}.yaml",
                "MODEL.WEIGHTS",
                subprocess.check_output(f"ls weights/trained/open-vocabulary/lvis/vit{vit}_*.pth | head -n 1",
                                        shell=True).strip(),
                "DE.OFFLINE_RPN_CONFIG", "configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
                "OUTPUT_DIR", f"output/eval/open-vocabulary/lvis/vit{vit}/"
            ]
    elif task == "fsod":
        cmd = [
            "python3", "tools/train_net.py",
            "--num-gpus", str(num_gpus), "--eval-only",
            "--config-file", f"configs/few-shot/vit{vit}_shot{shot}.yaml",
            "MODEL.WEIGHTS",
            subprocess.check_output(f"ls weights/trained/few-shot/vit{vit}_*.pth | head -n 1", shell=True).strip(),
            "DE.OFFLINE_RPN_CONFIG", "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml",
            "OUTPUT_DIR", f"output/eval/few-shot/shot-{shot}/vit{vit}/"
        ]
    elif task == "osod":
        cmd = [
            "python3", "tools/train_net.py",
            "--num-gpus", str(num_gpus), "--eval-only",
            "--config-file", f"configs/one-shot/split{split}_vit{vit}.yaml",
            "MODEL.WEIGHTS",
            subprocess.check_output(f"ls weights/trained/one-shot/vit{vit}_*.split{split}.pth | head -n 1",
                                    shell=True).strip(),
            "DE.OFFLINE_RPN_CONFIG", "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml",
            "OUTPUT_DIR", f"output/eval/one-shot/split{split}/vit{vit}/",
            "DE.ONE_SHOT_MODE", "True"
        ]
    else:
        print("skip")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train_net.py with specified parameters.")
    parser.add_argument("--task", default="ovd", choices=["ovd", "fsod", "osod"], help="Task to execute")
    parser.add_argument("--vit", default="l", choices=["s", "b", "l"], help="Vit type")
    parser.add_argument("--dataset", default="coco", choices=["coco", "lvis"], help="Dataset to use")
    parser.add_argument("--shot", type=int, default=10, help="Shot parameter")
    parser.add_argument("--split", type=int, default=1, help="Split parameter")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use")

    args = parser.parse_args()

    try:
        pynvml.nvmlInit()  # Initialize NVML
        num_gpus = args.num_gpus if args.num_gpus is not None else pynvml.nvmlDeviceGetCount()
        main(args.task, args.vit, args.dataset, args.shot, args.split, num_gpus)
    except pynvml.NVMLError as e:
        print(f"Error: {e}")
    finally:
        pynvml.nvmlShutdown()  # Shutdown NVML
