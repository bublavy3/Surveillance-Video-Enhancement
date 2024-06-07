import cv2
from cv2 import dnn_superres
import ctypes
from datetime import datetime
from deoldify.visualize import *
from enum import Enum
import numpy as np
from pathlib import Path
from PIL import Image
import time
import torch
import warnings

from live_video_acquirement import LiveVideo, StreamSource

class UpscaleModel(Enum):
    EDSR = 'EDSR'
    ESPCN = 'ESPCN'
    FSRCNN = 'FSRCNN'
    LapSRN = 'LapSRN'


class SurveillanceVideo:
    def __init__(self, original_sample_file: str = None):
        """
        Sets up class attributes according to system attributes and default
        settings, stores default video and according properties if provided.

        Keyword Args:
            original_sample_file (str):  video file address (default None)
        """

        if original_sample_file is not None:
            self.original_filename = original_sample_file
            self.format_identifier = original_sample_file.split('.')[-1]
            self.original_video = cv2.VideoCapture(original_sample_file)
            self.fps = self.original_video.get(cv2.CAP_PROP_FPS)

            self.fourcc = int(self.original_video.get(cv2.CAP_PROP_FOURCC))
        else:
            self.fps = 30

        self.period = int(1000 / self.fps)
        user32 = ctypes.windll.user32
        self.available_screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        self.method_register = {'BRT': self.frame_brighten,
                                'EQH': self.frame_equalize_histogram,
                                'SRP': self.frame_sharpen,
                                'SRS': self.frame_sharpen_subtract_smoothened,
                                'DNS': self.frame_denoise,
                                'DBL': self.frame_deblur}
        self.settings = {'progress_prints': False,
                        'time_prints': False,
                        'time_print_rounding': None,
                        'time_print_frames': True,
                        'time_print_resolution': False,
                        'sharpen_kernel': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                        'sharpen_negative_blur_kernel_size': 3,
                        'brighten_factor': 50,
                        'equalize_histogram_in_RGB': True,
                        'histogram_equalization_clip_limit': 40,
                        'histogram_equalization_tile_size': 8,
                        'denoise_strength': 3,
                        'denoise_window_size': 7,
                        'denoise_search_size': 21,
                        'deblur_kernel_size': 7,
                        'deblur_kernel_sigma': 5,
                        'wiener_factor_constant': 0.5,
                        'deblur_equalize_histogram': True,
                        'binary_threshold': 150,
                        'recolorize_render_factor': 21}

    def time_print(self, process_name: str, time_elapsed: float, frames: int,
                    resolution_tuple: tuple, short: bool = False):
        """
        Prints time used by process in specified format.

        Args:
            process_name (str): process name to be printed
            time_elapsed (float): time measured
            frames (int): number of frames on which was measured process applied
            resolution_tuple (tuple): tuple of 2 values (int) representing width
                                        and height of video
            short (bool): wether to use shorter, briefer form of report
        """
        rounding = self.settings['time_print_rounding']
        if rounding is not None:
            time_elapsed = round(time_elapsed, rounding)

        if self.settings['time_print_frames']:
            if short:
                frames = f", {frames} frames "
            else:
                frames = f" for {frames} frames"
        else:
            frames = ""

        if self.settings['time_print_resolution']:
            if short:
                resolution = f", {resolution_tuple[0]}x{resolution_tuple[1]}px "
            else:
                resolution = f" in {resolution_tuple[0]}x{resolution_tuple[1]} px"
        else:
            resolution = ""

        if short:
            print(f"{process_name}: {time_elapsed}s{frames}{resolution}")
        else:
            print(f"{process_name} took {time_elapsed} seconds{frames}{resolution}")

    def play_one(self, video, interpolation=cv2.INTER_CUBIC, savename: str = None):
        """
        Plays one video.

        Args:
            video (cv2.VideoCapture): video to be played
            interpolation (cv2.InterpolationFlags): interpolation used to scale
                                                    video to fit size to the more
                                                    limitting screen dimension,
                                                    set None to keep original
                                                    video resolution instead
                                                    (default cv2.INTER_CUBIC)
            savename (str): file name under which to save video (default None)
        """
        video_width, video_height = int(video.get(3)), int(video.get(4))
        width_ratio, height_ratio = video_width / self.available_screensize[0], video_height / self.available_screensize[1]

        if savename:
            format_identifier = 'mp4'
            filename = f'{savename}.{format_identifier}'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            savefile = cv2.VideoWriter(filename, fourcc, self.fps, (video_width, video_height))

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            if savename:
                savefile.write(frame)
            if interpolation is not None:
                if width_ratio > height_ratio:
                    frame = cv2.resize(frame,(self.available_screensize[0], int(video_height / width_ratio)), interpolation=interpolation)
                else:
                    frame = cv2.resize(frame,(int(video_width / height_ratio), self.available_screensize[1]), interpolation=interpolation)
            cv2.imshow("video - press ESC to close", frame)
            key = cv2.waitKey(self.period)
            if key == 27:
                break
        cv2.destroyAllWindows()
        if savename:
            savefile.release()

    def play_two(self, video1, video2, interpolation=cv2.INTER_CUBIC, savename: str = None):
        """
        Plays two videos.

        Args:
            video1 (cv2.VideoCapture): first video to be played
            video2 (cv2.VideoCapture): second video to be played
            interpolation (cv2.InterpolationFlags): interpolation used to scale
                                                    videos to fit size to the more
                                                    limitting screen dimension,
                                                    (default cv2.INTER_CUBIC)
            savename (str): file name under which to save collage (default None)
        """
        video_width1, video_height1 = int(video1.get(3)), int(video1.get(4))
        video_width2, video_height2 = int(video2.get(3)), int(video2.get(4))
        if round(video_width1 / video_height1, 3) != round(video_width2 / video_height2, 3):
            raise ValueError("Videos must have same width to height ratio")
        w_h_ratio = video_width1 / video_height1
        if w_h_ratio > self.available_screensize[0] / self.available_screensize[1]:
            concat_method = cv2.vconcat
            window_height = self.available_screensize[1]
            window_width = int((self.available_screensize[1] // 2) * w_h_ratio)
            video_height, video_width = window_height // 2, window_width
        else:
            concat_method = cv2.hconcat
            window_width = self.available_screensize[0]
            window_height = int((self.available_screensize[0] // 2) / w_h_ratio)
            video_width, video_height = window_width // 2, window_height

        if savename:
            format_identifier = 'mp4'
            filename = f'{savename}.{format_identifier}'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            savefile = cv2.VideoWriter(filename, fourcc, self.fps, (window_width, window_height))

        while True:
            hasFrame1, frame1 = video1.read()
            hasFrame2, frame2 = video2.read()
            if not hasFrame1 or not hasFrame2:
                break
            frame1 = cv2.resize(frame1, (video_width, video_height), interpolation=interpolation)
            frame2 = cv2.resize(frame2, (video_width, video_height), interpolation=interpolation)
            concat_frames = concat_method([frame1, frame2])
            cv2.imshow("comparison  - press ESC to close", concat_frames)
            if savename:
                savefile.write(concat_frames)
            key = cv2.waitKey(self.period)
            if key == 27:
                break
        cv2.destroyAllWindows()
        if savename:
            savefile.release()

    def play_two_manual_size(self, video1, video2, video_width: int, video_height: int,
                            interpolation1=None, interpolation2=None, concat_method=cv2.hconcat,
                            fps: int = None, savename: str = None):
        """
        Plays two videos scaled to given size by given interpolations.

        Args:
            video1 (cv2.VideoCapture): first video to be played
            video2 (cv2.VideoCapture): second video to be played
            video_width (int): width to which each video shall by scaled
            video_height (int): height to which each video shall by scaled
            interpolation1 (cv2.InterpolationFlags): interpolation to be
                                    used to scale 1st video to fit size to
                                    the more limitting screen dimension
                                    (default is None if same size, otherwise
                                    cv2.INTER_CUBIC)
            interpolation2 (cv2.InterpolationFlags): interpolation to be
                                    used to scale 2nd video to fit size to
                                    the more limitting screen dimension
                                    (default is None if same size, otherwise
                                    cv2.INTER_CUBIC)
            concat_method (cv2.hconcat / cv2.vconcat): concatenation method
            fps (int): frames per second of video collage (default class default)
            savename (str): file name under which to save collage (default None)
        """
        video1_width, video1_height = int(video1.get(3)), int(video1.get(4))
        video2_width, video2_height = int(video2.get(3)), int(video2.get(4))
        if interpolation1 is None and (video1_width != video_width or video1_height != video_height):
            interpolation1 = cv2.INTER_CUBIC
        if interpolation2 is None and (video1_width != video_width or video1_height != video_height):
            interpolation2 = cv2.INTER_CUBIC

        if fps is None:
            fps = self.fps
            period = self.period
        else:
            period = int(1000 / fps)

        if savename:
            format_identifier = 'mp4'
            filename = f'{savename}.{format_identifier}'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if concat_method == cv2.vconcat:
                collage_width = video_width
                collage_height = video_height * 2
            else:
                collage_width = video_width * 2
                collage_height = video_height
            savefile = cv2.VideoWriter(filename, fourcc, fps, (collage_width, collage_height))

        while True:
            hasFrame1, frame1 = video1.read()
            hasFrame2, frame2 = video2.read()
            if not hasFrame1 or not hasFrame2:
                break
            if interpolation1 is not None:
                frame1 = cv2.resize(frame1, (video_width, video_height), interpolation=interpolation1)
            if interpolation2 is not None:
                frame2 = cv2.resize(frame2, (video_width, video_height), interpolation=interpolation2)

            concat_frames = concat_method([frame1, frame2])
            cv2.imshow("comparison - press ESC to close", concat_frames)
            if savename:
                savefile.write(concat_frames)
            key = cv2.waitKey(period)
            if key == 27:
                break

        cv2.destroyAllWindows()
        if savename:
            savefile.release()

    def play_four(self, video1, video2, video3, video4, interpolation=cv2.INTER_CUBIC, savename: str = None):
        """
        Plays four videos.

        Args:
            video1 (cv2.VideoCapture): 1st video to be played
            video2 (cv2.VideoCapture): 2nd video to be played
            video3 (cv2.VideoCapture): 3rd video to be played
            video4 (cv2.VideoCapture): 4th video to be played
            interpolation (cv2.InterpolationFlags): interpolation used to scale
                                                    videos to fit size to the more
                                                    limitting screen dimension,
                                                    (default cv2.INTER_CUBIC)
            savename (str): file name under which to save collage (default None)
        """
        video_width1, video_height1 = int(video1.get(3)), int(video1.get(4))
        video_width2, video_height2 = int(video2.get(3)), int(video2.get(4))
        video_width3, video_height3 = int(video3.get(3)), int(video3.get(4))
        if video4 is not None:
            video_width4, video_height4 = int(video4.get(3)), int(video4.get(4))
        w_h_ratio = video_width1 / video_height1
        if (round(w_h_ratio, 3) != round(video_width2 / video_height2, 3) or
            round(w_h_ratio, 3) != round(video_width3 / video_height3, 3) or
            video4 is not None and round(w_h_ratio, 3) != round(video_width4 / video_height4, 3)):
            raise ValueError("Videos must have same width to height ratio")
        if w_h_ratio < self.available_screensize[0] / self.available_screensize[1]:
            window_height = self.available_screensize[1]
            window_width = int(self.available_screensize[1] * w_h_ratio)
            video_height, video_width = window_height // 2, window_width // 2
        else:
            window_width = self.available_screensize[0]
            window_height = int(self.available_screensize[0] / w_h_ratio)
            video_width, video_height = window_width // 2, window_height // 2

        if savename:
            format_identifier = 'mp4'
            filename = f'{savename}.{format_identifier}'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            savefile = cv2.VideoWriter(filename, fourcc, self.fps, (window_width, window_height))

        if video4 is None:
            frame4 = np.zeros((video_height, video_width, 3), dtype=np.uint8)   # black image to show instead
            hasFrame4 = True

        while True:
            hasFrame1, frame1 = video1.read()
            hasFrame2, frame2 = video2.read()
            hasFrame3, frame3 = video3.read()
            if video4 is not None:
                hasFrame4, frame4 = video4.read()

            if not hasFrame1 or not hasFrame2 or not hasFrame3 or not hasFrame4:
                break
            frame1 = cv2.resize(frame1, (video_width, video_height), interpolation=interpolation)
            frame2 = cv2.resize(frame2, (video_width, video_height), interpolation=interpolation)
            frame3 = cv2.resize(frame3, (video_width, video_height), interpolation=interpolation)
            if video4 is not None:
                frame4 = cv2.resize(frame4, (video_width, video_height), interpolation=interpolation)
            concat_frames = cv2.vconcat([cv2.hconcat([frame1, frame2]), cv2.hconcat([frame3, frame4])])
            cv2.imshow("comparison  - press ESC to close", concat_frames)
            if savename:
                savefile.write(concat_frames)
            key = cv2.waitKey(self.period)
            if key == 27:
                break
        cv2.destroyAllWindows()
        if savename:
            savefile.release()

    def play_four_manual_size(self, video_list: list, video_width: int, video_height: int,
                            interpolation_list: list = None, fps: int = None, savename: str = None):
        """
        Plays four videos scaled to given size by given interpolations.

        Args:
            video_list (list): list of videos (cv2.VideoCapture) to be played
            video_width (int): width to which every video shall by scaled
            video_height (int): height to which every video shall by scaled
            interpolation_list (list): list of interpolations (cv2.InterpolationFlags)
                                        used to scale videos at the same index
                                        to fit size to the more limitting screen
                                        dimension (defaults are cv2.INTER_CUBIC)
            fps (int): frames per second of video collage (default class default)
            savename (str): file name under which to save collage (default None)
        """
        if interpolation_list is None:
            interpolation_list = [cv2.INTER_CUBIC for _ in range(4)]
        for i in range(0, 4):
            video = video_list[i]
            video_width_i, video_height_i = int(video.get(3)), int(video.get(4))
            if video_width_i == video_width and video_height_i == video_height:
                interpolation_list[i] = None

        if fps is None:
            fps = self.fps
            period = self.period
        else:
            period = int(1000 / fps)

        if savename:
            format_identifier = 'mp4'
            filename = f'{savename}.{format_identifier}'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            savefile = cv2.VideoWriter(filename, fourcc, fps, (video_width * 2, video_height * 2))

        frames = [None for _ in range(4)]
        while True:
            end = False
            for i in range(4):
                hasFrame, frame = video_list[i].read()
                if not hasFrame:
                    end = True
                    break
                if interpolation_list[i] is None:
                    frames[i] = frame
                else:
                    frames[i] = cv2.resize(frame, (video_width, video_height), interpolation=interpolation_list[i])
            if end:
                break

            concat_frames = cv2.vconcat([cv2.hconcat([frames[0], frames[1]]), cv2.hconcat([frames[2], frames[3]])])
            cv2.imshow("comparison - press ESC to close", concat_frames)
            if savename:
                savefile.write(concat_frames)
            key = cv2.waitKey(period)
            if key == 27:
                break

        cv2.destroyAllWindows()
        if savename:
            savefile.release()

    def play(self, video1, video2=None, video3=None, video4=None, interpolation=cv2.INTER_CUBIC, savename: str = None):
        """
        Plays up to 4 videos in automatically laid out collage, interpolated to
        fit size to the more limitting screen dimension.

        Args:
            video1 (cv2.VideoCapture): 1st video to be played
            video2 (cv2.VideoCapture): 2nd video to be played (default None)
            video3 (cv2.VideoCapture): 3rd video to be played (default None)
            video4 (cv2.VideoCapture): 4th video to be played (default None)
            interpolation (cv2.InterpolationFlags): interpolation used to scale
                                                    videos (default cv2.INTER_CUBIC)
            savename (str): file name under which to save collage (default None)
        """
        if video3:
            self.play_four(video1, video2, video3, video4, interpolation, savename)
        elif video2:
            self.play_two(video1, video2, interpolation, savename)
        elif video1:
            self.play_one(video1, interpolation, savename)
        else:
            raise ValueError("Video 1 is missing")

    def brighten(self, video, factor: int = None, return_filename: bool = False):
        """
        Brightens the video globally by a value of factor, while ensuring
        there is no intensity overflow.

        Args:
            video (cv2.VideoCapture): input video
            factor (int): intensity value by which to brighten video (default 50)
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'brightened_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        frame_count = 0
        start_time = time.time()

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame = self.frame_brighten(frame, factor)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Brightening", elapsed_time, frames=frame_count, resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def equalize_histogram(self, video, return_filename: bool = False):
        """
        Equalizes histogram of video in YCrCb format.

        Args:
            video (cv2.VideoCapture): input video
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'enhanced_histogram_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        start_time = time.time()
        frame_count = 0

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            channels = list(cv2.split(frame_ycrcb))
            channels[0] = cv2.equalizeHist(channels[0])
            frame_ycrcb = cv2.merge(channels)
            frame = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Histogram equalization", elapsed_time, frames=frame_count, resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def equalize_histogram_adaptively(self, video, clip_limit: int = None, tile_size: int = None, return_filename: bool = None):
        """
        Equalizes histogram of video in YCrCb format adaptively.

        Args:
            video (cv2.VideoCapture): input video
            clip_limit (int): threshold for contrast limiting (default 40)
            tile_size (int): size of tiles to equalize histogram in (default 8)
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'enhanced_histogram_adaptively_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')    # self.fourcc did not work
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        if clip_limit is None:
            clip_limit = self.settings['histogram_equalization_clip_limit']
        if tile_size is None:
            tile_size = self.settings['histogram_equalization_tile_size']
        clahe = cv2.createCLAHE(clip_limit, (tile_size, tile_size))

        start_time = time.time()
        frame_count = 0

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            channels = list(cv2.split(frame_ycrcb))
            channels[0] = clahe.apply(channels[0])
            frame_ycrcb = cv2.merge(channels)
            frame = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Adaptive histogram equalization", elapsed_time, frames=frame_count,
                            resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def sharpen(self, video, kernel=None, return_filename: bool = False):
        """
        Applies laplacian edge filter to sharpen the video.

        Args:
            video (cv2.VideoCapture): input video
            kernel (list): list of (list of (int)) representing kernel to apply
                            onto pixels in order to sharpen edges (default
                            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        if kernel is None:
            kernel = self.settings['sharpen_kernel']
        kernel_np = np.array(kernel)
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'sharpened_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')    # self.fourcc did not work
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        start_time = time.time()
        frame_count = 0

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame = cv2.filter2D(frame, -1, kernel_np)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Sharpening", elapsed_time, frames=frame_count, resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def sharpen_subtract_smoothened(self, frame, kernel_size: int = None):
        """
        Applies sharpening on video by subtracting the gaussian blurred image
        of each frame.

        Args:
            video (cv2.VideoCapture): input video
            kernel_size (int): size of kernel to calculate standard deviations
                                from
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'sharpened_subtracting_smoothened_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        frame_count = 0
        start_time = time.time()

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame = self.sharpen_subtract_smoothened(frame, kernel_size)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Sharpening by subtracting smoothened image", elapsed_time, frames=frame_count,
                            resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def denoise(self, video, reference_frames_around: int = 5, return_filename: bool = False):
        """
        Applies Non-Local Means Denoising algorithm using multiple frame reference
        effectively removing Gaussian noise from video while preserving edges.

        Args:
            video (cv2.VideoCapture): input video
            reference_frames_around (int): size of sliding window of frames to
                                            be used as reference for denoising
                                            (default 5)
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'denoised_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')    # self.fourcc did not work
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        frames = []
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frames.append(frame)

        total_frames = len(frames)
        start_time = time.time()

        for i in range(0, total_frames):
            if self.settings['progress_prints']:
                print(f"Denoising frame {i+1}/{total_frames}")
            if i >= reference_frames_around // 2 and i < total_frames - reference_frames_around // 2:
                frame_dns = cv2.fastNlMeansDenoisingColoredMulti(srcImgs=frames, imgToDenoiseIndex=i,
                temporalWindowSize=reference_frames_around, h=self.settings['denoise_strength'],
                hColor=self.settings['denoise_strength'], templateWindowSize=self.settings['denoise_window_size'],
                searchWindowSize=self.settings['denoise_search_size'])
            else:
                frame_dns = self.frame_denoise(frames[i])
            enhanced.write(frame_dns)

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Denoising", elapsed_time, frames=total_frames, resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def denoise_by_frame(self, video, return_filename: bool = False):
        """
        Applies Non-Local Means Denoising algorithm treating each frame individually
        (to increase speed) to remove Gaussian noise from video while preserving
        edges.

        Args:
            video (cv2.VideoCapture): input video
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'denoised_by_frame_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        start_time = time.time()

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            if self.settings['progress_prints']:
                print(f"Denoising frame {frame_count}/{total_frames}")
            frame = self.frame_denoise(frame)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Denoising", elapsed_time, frames=frame_count, resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def deblur(self, video, kernel_size: int = None, kernel_sigma: int = None, wiener_const: float = None, eq_histogram: bool = None):
        """
        Deblur video by deconvolution with Gaussian blur filter.

        Args:
            video (cv2.VideoCapture): input video
            kernel_size (int): size of kernel for gaussian blur filter (default 7)
            kernel_sigma (int): sigma value of kernel for gaussian blur filter
                                (default 5)
            wiener_const (float): wiener constant to use in computation of
                                    wiener factor by which the image is multiplied
                                    to compensate for noise (default 0.5)
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        if eq_histogram is None:
            eq_histogram = self.settings['deblur_equalize_histogram']
        if eq_histogram:
            savename = f'equalized_deblurred_{current_time}'
            process_name = "Deblurring and equalizing histogram"
        else:
            savename = f'deblurred_{current_time}'
            process_name = "Deblurring"
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        frame_count = 0
        start_time = time.time()

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame = self.frame_deblur(frame, kernel_size, kernel_sigma, wiener_const, eq_histogram)
            enhanced.write(frame)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print(process_name, elapsed_time, frames=frame_count,
                            resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def dnn_upscale(self, video, scale: int, model: UpscaleModel, return_filename: bool = False):
        """
        Upscales video using pre-trained deep neural network model.

        Args:
            video (cv2.VideoCapture): input video
            scale (int): multiply of size that's desired
            model (UpscaleModel): pre-trained model to use
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: upscaled video OR str: upscaled video file address
        """
        model = model.value
        sr = dnn_superres.DnnSuperResImpl_create()
        path = 'dnn_models/' + model + '_x' + str(scale) + '.pb'
        sr.readModel(path)
        sr.setModel(model.lower(), scale)

        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        savename = f'upscaled_{model}_x{str(scale)}_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)) * scale, int(video.get(4)) * scale))

        start_time = time.time()
        frame_count = 0

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame = sr.upsample(frame)
            enhanced.write(frame)
            if self.settings['progress_prints']:
                print("frame " + str(frame_count) + " upscaled by " + model)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print(f"DNN upscaling by {model}", elapsed_time, frames=frame_count,
                            resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def cutout(self, video, x_from: int, x_to: int, y_from: int, y_to: int, return_filename: bool = False):
        """
        Crops video.

        Args:
            video (cv2.VideoCapture): input video
            x_from(int): left border of crop region
            x_to(int): right border of crop region
            y_from(int): upper border of crop region
            y_to(int): lower border of crop region
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: cropped video OR str: cropped video file address
        """
        format_identifier = 'mp4'
        file = 'cutout.' + format_identifier
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (x_to - x_from, y_to - y_from))
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            frame = frame[y_from:y_to, x_from:x_to]
            enhanced.write(frame)
        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def frame_brighten(self, frame, factor: int = None):
        """
        Brightens the frame globally by a value of factor, while ensuring
        there is no intensity overflow.

        Args:
            frame (numpy.ndarray): input frame
            factor (int): intensity value by which to brighten frame (default 50)

        Returns:
            (numpy.ndarray): enhanced frame
        """
        if factor is None:
            factor = self.settings['brighten_factor']
        return np.clip(frame.astype(int) + factor, 0, 255).astype(np.uint8)

    def frame_equalize_histogram(self, frame, color: bool = None):
        """
        Equalizes histogram of frame in YCrCb format.

        Args:
            frame (numpy.ndarray): input frame
            color (int): wether to equalize histogram of RGB-channeled image
                        (default True)

        Returns:
            (numpy.ndarray): enhanced frame
        """
        if color is None:
            color = self.settings['equalize_histogram_in_RGB']
        if color:
            frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            channels = list(cv2.split(frame_ycrcb))
            channels[0] = cv2.equalizeHist(channels[0])
            frame_ycrcb = cv2.merge(channels)
            return cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(frame_gray)
            return cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    def frame_sharpen(self, frame, kernel=None):
        """
        Applies laplacian edge filter to sharpen the frame.

        Args:
            frame (numpy.ndarray): input frame
            kernel (list): list of (list of (int)) representing kernel to apply
                            onto pixels in order to sharpen edges (default
                            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        Returns:
            (numpy.ndarray): enhanced frame
        """
        if kernel is None:
            kernel = self.settings['sharpen_kernel']
        return cv2.filter2D(frame, -1, np.array(kernel))

    def frame_sharpen_subtract_smoothened(self, frame, kernel_size: int = None):
        """
        Applies sharpening by subtracting the gaussian blurred image.

        Args:
            frame (numpy.ndarray): input frame
            kernel_size (int): size of kernel to calculate standard deviations
                                from

        Returns:
            (numpy.ndarray): enhanced frame
        """
        if kernel_size is None:
            kernel_size = self.settings['sharpen_negative_blur_kernel_size']
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0, 0)
        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    def frame_denoise(self, frame, strength: int = None, window_size: int = None, search_size: int = None):
        """
        Applies Non-Local Means Denoising algorithm effectively removing
        Gaussian noise from frame while preserving edges.

        Args:
            frame (numpy.ndarray): input frame
            strength (int): filter strength for luminance component, bigger
                            value removes more noise but also removes more image
                            details (default 3)
            window_size (int): size of window from which weights are computed
                                (default 7)
            search_size (int): size of window within which we search for average
                                value of pixel (default 21)

        Returns:
            (numpy.ndarray): enhanced frame
        """
        if strength is None:
            strength = self.settings['denoise_strength']
        if window_size is None:
            window_size = self.settings['denoise_window_size']
        if search_size is None:
            window_size = self.settings['denoise_search_size']
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, window_size, search_size)

    def _gaussian_filter(self, kernel_size, img,sigma=1, muu=0):
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                           np.linspace(-1, 1, kernel_size))
        dst = np.sqrt(x**2 + y**2)
        normal = 1/ (((2 * np.pi)**0.5) * sigma)
        gauss = np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal
        gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
        return gauss

    def _fft_deblur(self, img, kernel_size, kernel_sigma=5, wiener_const=0.5):
        gauss = self._gaussian_filter(kernel_size, img, kernel_sigma)
        img_fft = np.fft.fft2(img)
        gauss_fft = np.fft.fft2(gauss)
        wiener_factor = 1 / (1 + (wiener_const / np.abs(gauss_fft)**2))
        recon = img_fft / gauss_fft
        recon *= wiener_factor
        recon = np.abs(np.fft.ifft2(recon))
        return recon

    def frame_deblur(self, frame, kernel_size: int = None, kernel_sigma: int = None, wiener_const: float = None, eq_histogram: bool = None):
        """
        Deblurs frame by deconvolution with Gaussian blur filter.

        Args:
            frame (numpy.ndarray): input frame
            kernel_size (int): size of kernel for gaussian blur filter (default 7)
            kernel_sigma (int): sigma value of kernel for gaussian blur filter
                                (default 5)
            wiener_const (float): wiener constant to use in computation of
                                    wiener factor by which the image is multiplied

        Returns:
            (numpy.ndarray): enhanced frame
        """
        if kernel_size is None:
            kernel_size = self.settings['deblur_kernel_size']
        if kernel_sigma is None:
            kernel_sigma = self.settings['deblur_kernel_sigma']
        if wiener_const is None:
            wiener_const = self.settings['wiener_factor_constant']
        if eq_histogram is None:
            eq_histogram = self.settings['deblur_equalize_histogram']
        channels = list(cv2.split(frame))
        for channel in range(min(len(channels), 3)):
            channels[channel] = self._fft_deblur(channels[channel], kernel_size, kernel_sigma, wiener_const)
        deblurred = cv2.merge(channels)
        deblurred = deblurred.astype(np.uint8)
        if eq_histogram:
            deblurred = self.frame_equalize_histogram(deblurred)
        return deblurred

    def combiner(self, video, method_list: list, savename: str = None, return_filename: bool = False):
        """
        Applies listed methods on video frame by frame, enhancing each frame
        fully before moving onto next one.

        Args:
            video (cv2.VideoCapture): input video
            method_list (list): sequence of 3-letter codes of methods to apply
            savename (str): file name under which to save video (default None)
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        if savename is None:
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            savename = f'combined_{current_time}'
        format_identifier = 'mp4'
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        enhanced = cv2.VideoWriter(file, fourcc, self.fps, (int(video.get(3)), int(video.get(4))))

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        index = 1
        start_time = time.time()

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            for method in method_list:
                if self.settings['progress_prints']:
                    print(f"Applying {method} on frame {index}/{total_frames} of video {savename}")
                frame = self.method_register[method](frame)
            enhanced.write(frame)
            index += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            methods = ', '.join(method_list)
            self.time_print(f"Applying {methods}", elapsed_time,
            frames=total_frames, resolution_tuple=(int(video.get(3)), int(video.get(4))))

        enhanced.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def track_movement_thermo_to_rgb(self, video_rgb, video_thermo, return_filename: bool = False):
        """
        Maps movement detected in infrared (thermo) video onto RGB video of same
        scene, highlighting silhouettes of moving objects with temperature
        different from their surroundings (especially people). Follow the commands
        in displayed window titles to enable correct pairing between scene
        projection in RGB and in thermo recording.

        Args:
            video_rgb (cv2.VideoCapture): input RGB video of the scene
            video_thermo (cv2.VideoCapture): input infra / thermo video of the scene
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: RGB video complemented by silhouttes
            OR str: its file address
        """
        corner_points = []

        def mouseHandler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK and len(corner_points) < 4:
                corner_points.append((x, y))

        title_guide = ("mark with dbclck: 1) point A in thermo, 2) point A in RGB, 3) point B in thermo, 4) point B in RGB,"
                            " press any key to continue")

        hasFrame_rgb, frame_rgb = video_rgb.read()
        hasFrame_thermo, frame_thermo = video_thermo.read()
        height_rgb, width_rgb, _ = frame_rgb.shape
        height_thermo, width_thermo, _ = frame_thermo.shape
        video_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_thermo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        canvas_height = max(height_rgb, height_thermo)
        canvas_width = width_rgb + width_thermo
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:height_rgb, :width_rgb] = frame_rgb
        canvas[:height_thermo, width_rgb:] = frame_thermo

        canvas_aspect_ratio = canvas_width / canvas_height
        if canvas_aspect_ratio > (self.available_screensize[0] / self.available_screensize[1]):
            new_canvas_width = self.available_screensize[0]
            new_canvas_height = int(self.available_screensize[0] / canvas_aspect_ratio)
        else:
            new_canvas_height = self.available_screensize[1]
            new_canvas_width = int(self.available_screensize[1] * canvas_aspect_ratio)
        canvas_resized = cv2.resize(canvas, (new_canvas_width, new_canvas_height))
        resizing_factor = new_canvas_width / canvas_width

        aligned = False
        while not aligned:
            del corner_points[:]
            cv2.namedWindow(title_guide)
            cv2.setMouseCallback(title_guide, mouseHandler)
            cv2.imshow(title_guide, canvas_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if len(corner_points) != 4:
                raise AssertionError("Corner points were not selected properly")

            corner_A_thermo_absolute, corner_A_rgb, corner_B_thermo_absolute, corner_B_rgb = corner_points
            corner_A_thermo = (corner_A_thermo_absolute[0] - int(width_rgb * resizing_factor), corner_A_thermo_absolute[1])
            corner_B_thermo = (corner_B_thermo_absolute[0] - int(width_rgb * resizing_factor), corner_B_thermo_absolute[1])
            corner_hdist_thermo = abs(corner_A_thermo[0] - corner_B_thermo[0])
            corner_vdist_thermo = abs(corner_A_thermo[1] - corner_B_thermo[1])
            corner_hdist_rgb = abs(corner_A_rgb[0] - corner_B_rgb[0])
            corner_vdist_rgb = abs(corner_A_rgb[1] - corner_B_rgb[1])
            hscaling = corner_hdist_rgb / corner_hdist_thermo
            vscaling = corner_vdist_rgb / corner_vdist_thermo
            left_border_rgb = int(corner_A_rgb[0] - corner_A_thermo[0] * hscaling)
            right_border_rgb = int(left_border_rgb + width_thermo * resizing_factor * hscaling)
            top_border_rgb = int(corner_A_rgb[1] - corner_A_thermo[1] * vscaling)
            bottom_border_rgb = int(top_border_rgb + height_thermo * resizing_factor * vscaling)

            canvas_marked = canvas_resized.copy()
            cv2.rectangle(canvas_marked, (left_border_rgb, top_border_rgb), (right_border_rgb, bottom_border_rgb), (0, 0, 255), 3)
            cv2.imshow("confirm view alignment by pressing ENTER, or press ESC to repeat", canvas_marked)
            key = cv2.waitKey(0)
            if key == 13:
                print("Processing...")
                aligned = True
            cv2.destroyAllWindows()

        backSub = cv2.createBackgroundSubtractorMOG2()
        kernel = np.ones((3, 3), np.uint8)
        min_contour_area = 100

        fps_rgb = int(video_rgb.get(cv2.CAP_PROP_FPS))
        fps_thermo = int(video_thermo.get(cv2.CAP_PROP_FPS))
        frame_step_rgb = 1 / fps_rgb
        frame_step_thermo = 1 / fps_thermo
        time_rgb = 0
        time_thermo = 0

        format_identifier = 'mp4'
        file = 'motion_highlighted.' + format_identifier
        fourcc = cv2.VideoWriter_fourcc(*'XVID')    # self.fourcc did not work
        enhanced = cv2.VideoWriter(file, fourcc, fps_rgb, (int(width_rgb), int(height_rgb)))

        start_time = time.time()
        frame_count = 0

        while True:
            hasFrame_rgb, frame_rgb = video_rgb.read()
            if time_thermo <= time_rgb:
                hasFrame_thermo, frame_thermo = video_thermo.read()
                time_thermo += frame_step_thermo
            time_rgb += frame_step_rgb
            if not hasFrame_rgb or not hasFrame_thermo:
                break

            fgmask = backSub.apply(frame_thermo)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask_dilate = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
            contours_outer, _ = cv2.findContours(fgmask_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours_outer if cv2.contourArea(cnt) > min_contour_area]
            for contour_idx in range(len(contours)):
                cnt_points = contours[contour_idx]
                for point_coords_idx in range(len(cnt_points)):
                    cnt_points[point_coords_idx][0][0] = cnt_points[point_coords_idx][0][0] * hscaling + left_border_rgb / resizing_factor
                    cnt_points[point_coords_idx][0][1] = cnt_points[point_coords_idx][0][1] * vscaling + top_border_rgb / resizing_factor

            contour_color = (0, 0, 255)
            contoured_rgb = cv2.drawContours(frame_rgb, contours, -1, contour_color, 3) #cv2.FILLED for filling contoured areas
            enhanced.write(contoured_rgb)
            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Motion tracking", elapsed_time, frames=frame_count, resolution_tuple=(int(width_rgb), int(height_rgb)))

        enhanced.release()
        video_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_thermo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_rgb.release()
        video_thermo.release()

        if return_filename:
            return file
        return cv2.VideoCapture(file)

    def recolorize_video(self, video_filename: str, render_factor: int = None, return_filename: bool = False):
        """
        Recolorizes grayscale video using deep neural network from DeOldify
        project.

        Args:
            video_filename (str): file address of input video
            render_factor (int): demanded precision of recolorization (higher
                                    takes more time)
            return_filename (bool): wether to return enhanced video as file
                                    address instead of cv2.VideoCapture
                                    (default False)

        Returns:
            cv2.VideoCapture: enhanced video OR str: enhanced video file address
        """
        video = cv2.VideoCapture(video_filename)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        resolution_tuple = (int(video.get(3)), int(video.get(4)))
        video.release()

        video_colorizer = get_video_colorizer()
        if render_factor is None:
            render_factor = self.settings['recolorize_render_factor']

        start_time = time.time()

        video_path = video_colorizer.colorize_from_file_name(file_name=video_filename, render_factor=render_factor, watermarked=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.settings['time_prints']:
            self.time_print("Recolorization", elapsed_time, frames=total_frames, resolution_tuple=resolution_tuple)

        if return_filename:
            return str(video_path)
        return cv2.VideoCapture(str(video_path))

    def enhance_stream(self, destination: StreamSource, method_list: list, period: int = 1):
        """
        Applies listed methods on each frame read from source of stream
        and displays these frames in real time.

        Args:
            destination (StreamSource): input stream source
            method_list (list): sequence of 3-letter codes of methods to apply
            period (int): minimal amount of miliseconds between reading next
                            frame from source stream
        """
        stream = LiveVideo()
        video = stream.connect(destination)
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            for method in method_list:
                frame = self.method_register[method](frame)
            cv2.imshow("enhanced live stream - press ESC to close", frame)
            key = cv2.waitKey(period)
            if key == 27:
                break
        cv2.destroyAllWindows()
        video.release()


if __name__ == '__main__':
    # EXAMPLE USAGE:
    pass

##    path = 'sample_videos/student_street.mp4'
##    enhancer = SurveillanceVideo(path)
####    enhancer.settings["time_prints"] = True
####    enhancer.settings["time_print_rounding"] = 2
####    enhancer.settings["progress_prints"] = True
##    original = enhancer.original_video
##    equalized = enhancer.equalize_histogram(original)
##    sharpened = enhancer.sharpen(original)
##    equalized_sharpened = enhancer.equalize_histogram(sharpened)
##    enhancer.play(original, equalized, sharpened, equalized_sharpened, savename="student_street org, EQH, SRP, SRP+EQH")

##    path = 'sample_videos/dutch_street_sunset.mp4'
##    enhancer = SurveillanceVideo(path)
##    enhancer.settings["time_prints"] = True
##    enhancer.settings["time_print_rounding"] = 2
##    enhancer.settings["progress_prints"] = True
##    original = enhancer.original_video
##    adjusted = enhancer.combiner(original, ["DNS", "EQH", "SRP"], savename="dutch_street org, DNS+EQH+SRP")
##    enhancer.play(original, adjusted)

##    path = 'sample_videos/german_marketplace_eve.mp4'
##    enhancer = SurveillanceVideo(path)
####    enhancer.settings["time_prints"] = True
####    enhancer.settings["time_print_rounding"] = 2
####    enhancer.settings["progress_prints"] = True
##    original = enhancer.original_video
##    equalized_adaptively_v1 = enhancer.equalize_histogram_adaptively(original)
##    enhancer.settings["histogram_equalization_clip_limit"] = 20
##    enhancer.settings["histogram_equalization_tile_size"] = 4
##    equalized_adaptively_v2 = enhancer.equalize_histogram_adaptively(original)
##    enhancer.play(equalized_adaptively_v1, equalized_adaptively_v2)

##    path = 'sample_videos/student_street.mp4'
##    enhancer = SurveillanceVideo(path)
##    enhancer.settings["time_prints"] = True
##    enhancer.settings["time_print_rounding"] = 2
##    enhancer.settings["progress_prints"] = True
##    original = enhancer.original_video
##    cutout = enhancer.cutout(original, 350, 450, 250, 350)
##    cutout_to_interpolate = enhancer.cutout(original, 350, 450, 250, 350)
##    espcn = enhancer.dnn_upscale(cutout, 4, UpscaleModel.ESPCN)
##    fsrcnn = enhancer.dnn_upscale(cutout, 4, UpscaleModel.FSRCNN)
##    enhancer.play_four_manual_size([cutout, cutout_to_interpolate, espcn, fsrcnn], 400, 400,
##                                interpolation_list=[cv2.INTER_NEAREST, cv2.INTER_CUBIC, None, None],
##                                savename="scenarios/student_street/cutout org, bicubic, ESPCN, FSRCNN")

##    path = 'sample_videos/Hikvision_dark_distant_cut.mp4'
##    enhancer = SurveillanceVideo(path)
##    recolorized = enhancer.recolorize_video(enhancer.original_filename)
##    enhancer.play(recolorized)

##    path = 'sample_videos/Annke_RGB_dark.mp4'
##    enhancer = SurveillanceVideo(path)
####    enhancer.settings["time_prints"] = True
####    enhancer.settings["time_print_rounding"] = 2
####    enhancer.settings["progress_prints"] = True
##    original = enhancer.original_video
##    movement_tracked = enhancer.track_movement_thermo_to_rgb(original, cv2.VideoCapture('sample_videos/Annke_thermo_dark.mp4'))
##    enhancer.play(movement_tracked, savename="movement_tracking_dark")

##    enhancer = SurveillanceVideo()
##    enhancer.enhance_stream(StreamSource.AMERICAN_CROSSROAD, ["SRS"])