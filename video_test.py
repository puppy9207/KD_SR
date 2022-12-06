import os
import hydra
import cv2
import ffmpeg
import PIL.Image as pil_image
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models import Generator
from utils import preprocess, postprocess, get_concat_h, yuv2rgb, rgb2yuv, AverageMeter
from metrics import PSNR, SSIM, ERQA

@hydra.main(config_path="./configs", config_name="test")
def main(cfg):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Generator(cfg.test.generator).to(device)
    try:
        model.load_state_dict(torch.load(cfg.test.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(cfg.test.weights_file, map_location=device)["g"].items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise RuntimeError("Error when loading a model")
    model.eval()

    file_name = cfg.test.video_file
    target_file_name = f'supremo_{cfg.test.video_file.split("/")[-1]}'

    if not os.path.isfile(file_name):
        raise FileNotFoundError("해당 경로에 파일이 존재하지 않습니다.")
    
    video_ext = (
        "mp4", "m4v", "mkv", "webm",
        "mov", "avi", "wmv", "mpg", "flv","m2t", "mxf","MXF"
    )

    if not file_name.endswith(video_ext):
        raise Exception("파일의 확장자가 Video 파일이 아닙니다.")    
    
    streams = ffmpeg.probe(file_name, select_streams="v")["streams"][0]
    denominator, nominator = streams["r_frame_rate"].split("/")
    fps = float(denominator) / float(nominator)
    width = int(streams["width"])
    height = int(streams["height"])
    target_width = width * cfg.test.generator.scale if not cfg.test.with_metric else width
    target_height = height * cfg.test.generator.scale if not cfg.test.with_metric else height
    vcodec = streams["codec_name"]
    pix_fmt = streams["pix_fmt"]
    
    in_process = (
        ffmpeg
        .input(file_name)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
        .run_async(pipe_stdout=True)
    )
    
    out_process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(target_width, target_height), r=fps)
            .output(ffmpeg.input(file_name).audio, target_file_name, pix_fmt=pix_fmt, acodec="aac",crf=cfg.test.crf, vcodec=vcodec) # loglevel="quiet"
            .overwrite_output() 
            .run_async(pipe_stdin=True)
        )

    psnr = PSNR()
    ssim = SSIM()
    erqa = ERQA()

    while True:

        in_bytes = in_process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        if cfg.test.with_metric:
            in_frame= cv2.resize(in_frame, (width//cfg.test.generator.scale, height//cfg.test.generator.scale))

        bic = cv2.resize(in_frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        lr = preprocess(in_frame).to(device)

        with torch.no_grad():
            preds = model(lr)
        preds = postprocess(preds)

        # """Metrics"""
        psnr(preds, bic)
        ssim(preds, bic)
        print(erqa(preds, bic))

        if cfg.test.merge:    
            preds = np.hstack((preds[:,target_width//4:target_width//4*3,:], bic[:,target_width//4:target_width//4*3,:]))
        out_process.stdin.write(preds.tobytes())
    
    print(psnr)
    print(ssim)
    
    in_process.stdout.close()
    out_process.stdin.close()
    out_process.wait()
    in_process.wait()


if __name__ == "__main__":
    main()