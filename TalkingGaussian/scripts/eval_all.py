import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import re


def _run(cmd, cwd=None, allow_fail=False):
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n[RUN] {printable}")
    ret = subprocess.call(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if ret != 0 and not allow_fail:
        raise SystemExit(ret)
    return ret


def _run_with_output(cmd, cwd=None, allow_fail=False):
    """
    运行子进程并捕获输出，用于从外部脚本中解析指标数值。
    """
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n[RUN] {printable}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode, proc.stdout or ""


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _extract_frames(py, video, out_dir, fps=None):
    cmd = [py, os.path.join("evaluation", "extract_frames.py"), video, out_dir]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    _run(cmd, cwd=_project_root())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run all evaluation metrics (frame metrics / NIQE / FID / LSE) via the scripts in evaluation/."
        )
    )

    # Inputs
    parser.add_argument("--pred-video", default=None, help="Pred/generated video path (for PSNR/SSIM/LPIPS, NIQE, FID)")
    parser.add_argument("--gt-video", default=None, help="GT/reference video path (for PSNR/SSIM/LPIPS, FID)")
    parser.add_argument(
        "--gen-videos-dir",
        default=None,
        help="Directory containing generated .mp4 files (for LSE presets)",
    )

    # What to run
    parser.add_argument("--frame", action="store_true", help="Run frame metrics (PSNR/SSIM, optional LPIPS) on pred/gt videos")
    parser.add_argument("--niqe", action="store_true", help="Run NIQE on frames extracted from pred video")
    parser.add_argument("--fid", action="store_true", help="Run FID on frames extracted from pred/gt videos")
    parser.add_argument("--lse", action="store_true", help="Run LSE-C/LSE-D using SyncNet/Wav2Lip scripts")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run everything possible given the provided inputs",
    )

    # Frame-metrics options
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS in frame metrics")
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone (only used when --lpips)",
    )

    # Frame extraction options
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS used when extracting frames for NIQE/FID (default: original)",
    )

    # FID options
    parser.add_argument(
        "--fid-backend",
        default="clean-fid",
        choices=["clean-fid", "torch-fidelity"],
        help="FID backend (recommended: clean-fid)",
    )

    # LSE options
    parser.add_argument(
        "--lse-preset",
        default="wav2lip-real",
        choices=["wav2lip-real", "wav2lip-lrs"],
        help="LSE preset to run via evaluation/eval_lse.py",
    )
    parser.add_argument(
        "--syncnet-dir",
        default=os.path.join(_project_root(), "third_party", "syncnet_python"),
        help="Path to syncnet_python repo (for LSE)",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp_dir/",
        help="tmp_dir passed to some LSE pipelines",
    )
    parser.add_argument(
        "--setup-lse",
        action="store_true",
        help="Run evaluation/setup_lse.sh before LSE",
    )

    # Output
    parser.add_argument(
        "--work-dir",
        default=os.path.join(_project_root(), "evaluation", "_work"),
        help="Working directory for extracted frames",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write a JSON summary (best-effort)",
    )

    args = parser.parse_args()

    # Resolve what to run
    run_frame = args.frame
    run_niqe = args.niqe
    run_fid = args.fid
    run_lse = args.lse

    if args.all or (not (run_frame or run_niqe or run_fid or run_lse)):
        # If --all OR nothing specified: run whatever makes sense with inputs
        run_frame = args.pred_video is not None and args.gt_video is not None
        run_niqe = args.pred_video is not None
        run_fid = args.pred_video is not None and args.gt_video is not None
        run_lse = args.gen_videos_dir is not None

    root = _project_root()
    py = sys.executable

    results = {}
    _ensure_dir(args.work_dir)

    # 1) Frame metrics
    if run_frame:
        if not args.pred_video or not args.gt_video:
            raise SystemExit("--frame requires --pred-video and --gt-video")
        cmd = [
            py,
            os.path.join(root, "evaluation", "eval_frame_metrics.py"),
            args.pred_video,
            args.gt_video,
        ]
        if args.lpips:
            cmd += ["--lpips", "--lpips-net", args.lpips_net]
        ret, out = _run_with_output(cmd, allow_fail=False)

        psnr_val = None
        ssim_val = None
        lpips_val = None
        if out:
            m = re.search(r"PSNR\s*=\s*([0-9.+-eE]+)", out)
            if m:
                psnr_val = float(m.group(1))
            m = re.search(r"SSIM\s*=\s*([0-9.+-eE]+)", out)
            if m:
                ssim_val = float(m.group(1))
            if args.lpips:
                m = re.search(r"LPIPS.*?=\s*([0-9.+-eE]+)", out)
                if m:
                    lpips_val = float(m.group(1))

        results["frame"] = {
            "lpips": bool(args.lpips),
            "lpips_net": args.lpips_net,
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "LPIPS": lpips_val,
        }

    # 2) NIQE (extract frames from pred)
    pred_frames_dir = None
    if run_niqe or run_fid:
        if not args.pred_video:
            raise SystemExit("--niqe/--fid require --pred-video")
        pred_frames_dir = os.path.join(args.work_dir, "pred_frames")
        if os.path.exists(pred_frames_dir):
            shutil.rmtree(pred_frames_dir)
        _extract_frames(py, args.pred_video, pred_frames_dir, fps=args.fps)

    if run_niqe:
        # 在专门的 tg_niqe 环境中计算 NIQE（该环境已安装 pyiqa），并解析数值
        niqe_script = os.path.join(root, "evaluation", "eval_niqe.py")
        cmd = f"conda run -n tg_niqe --no-capture-output python \"{niqe_script}\" \"{pred_frames_dir}\""
        ret, out = _run_with_output(cmd, allow_fail=True)

        niqe_val = None
        if out:
            m = re.search(r"NIQE\s*\(pyiqa\)\s*=\s*([0-9.+-eE]+)", out)
            if m:
                niqe_val = float(m.group(1))

        results["niqe"] = {"frames_dir": pred_frames_dir, "ok": ret == 0, "NIQE": niqe_val}
        if ret != 0:
            print(
                "[WARN] NIQE failed. Please ensure 'pyiqa' is installed (pip install pyiqa). "
                "Continuing with remaining metrics.",
                file=sys.stderr,
            )

    # 3) FID (extract frames for gt too)
    if run_fid:
        if not args.gt_video:
            raise SystemExit("--fid requires --gt-video")
        gt_frames_dir = os.path.join(args.work_dir, "gt_frames")
        if os.path.exists(gt_frames_dir):
            shutil.rmtree(gt_frames_dir)
        _extract_frames(py, args.gt_video, gt_frames_dir, fps=args.fps)

        # 在专门的 tg_eval 环境中计算 FID（该环境已安装 clean-fid / torch-fidelity），并解析数值
        fid_script = os.path.join(root, "evaluation", "eval_fid.py")
        cmd = (
            "conda run -n tg_eval --no-capture-output python "
            f"\"{fid_script}\" \"{pred_frames_dir}\" \"{gt_frames_dir}\" "
            f"--backend {args.fid_backend}"
        )
        ret, out = _run_with_output(cmd, allow_fail=False)

        fid_val = None
        if out:
            m = re.search(r"FID.*?=\s*([0-9.+-eE]+)", out)
            if m:
                fid_val = float(m.group(1))

        results["fid"] = {
            "backend": args.fid_backend,
            "gen_frames": pred_frames_dir,
            "gt_frames": gt_frames_dir,
            "FID": fid_val,
        }

    # 4) LSE
    if run_lse:
        if not args.gen_videos_dir:
            raise SystemExit("--lse requires --gen-videos-dir")
        if args.setup_lse:
            _run(["bash", os.path.join(root, "evaluation", "setup_lse.sh")], cwd=root)

        cmd = [
            py,
            os.path.join(root, "evaluation", "eval_lse.py"),
            args.gen_videos_dir,
            "--preset",
            args.lse_preset,
            "--syncnet-dir",
            args.syncnet_dir,
            "--tmp-dir",
            args.tmp_dir,
        ]
        ret, out = _run_with_output(cmd, allow_fail=True)

        lse_c = None
        lse_d = None

        # 优先从标准输出里解析
        if out:
            m = re.search(r"LSE-C.*?=\s*([0-9.+-eE]+)", out)
            if m:
                lse_c = float(m.group(1))
            m = re.search(r"LSE-D.*?=\s*([0-9.+-eE]+)", out)
            if m:
                lse_d = float(m.group(1))

        # 其次从 SyncNet 输出的 all_scores.txt 中解析（如果存在）
        scores_path = os.path.join(args.syncnet_dir, "all_scores.txt")
        if (lse_c is None or lse_d is None) and os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                text = f.read()
            if lse_c is None:
                m = re.search(r"LSE-C.*?=\s*([0-9.+-eE]+)", text)
                if m:
                    lse_c = float(m.group(1))
            if lse_d is None:
                m = re.search(r"LSE-D.*?=\s*([0-9.+-eE]+)", text)
                if m:
                    lse_d = float(m.group(1))

        results["lse"] = {
            "preset": args.lse_preset,
            "gen_videos_dir": args.gen_videos_dir,
            "syncnet_dir": args.syncnet_dir,
            "LSE-C": lse_c,
            "LSE-D": lse_d,
        }

    if args.json_out:
        _ensure_dir(os.path.dirname(os.path.abspath(args.json_out)) or ".")
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[WROTE] {args.json_out}")


if __name__ == "__main__":
    main()
