#!/usr/bin/env python
import numpy
numpy.seterr(all='raise')
from PIL import Image
from math import cos, sin, sqrt, pi
import colorsys
import time
import concurrent.futures
import argparse
import pathlib
import sys
import subprocess
import math

PRIMARY_COLOR = [52, 94, 183]
BACKGROUND_COLOR = (255,255,255,255)
TRANSPARENT = (0,0,0,0)

FPS = 30
DURATION = 4
FRAMES = 0
WIDTH = 100
WIDTH_2 = 50

DEFAULT_FRAME = {
    "p": 2,
    "q": 3.2,
    "r": 0,
    "v0_color": PRIMARY_COLOR,
    "v1_color": PRIMARY_COLOR,
    "v2_color": PRIMARY_COLOR,
    "v0_tolerance": 0.05,
    "v1_tolerance": 0.01,
    "v2_tolerance": 0.01,
    "max_iterations": 5,
}

KEYFRAMES = [
    {"t": 0,  "p": 2, "q": 2.1, "r": 35, "v0_tolerance": 0, "max_iterations": 1},
    # {"t": 5,  "q": 2.1, "r": 35, "v0_tolerance": 0.001, "max_iterations": 10},
    {"t": 25,  "v0_tolerance": 0.01, "max_iterations": 20},
    {"t": 75, "p": 2, "q": 2.55, "r": 1000, "v0_tolerance": 0.2, "max_iterations": 500},
    {"t": 95, "p": 2, "q": 3.1, "r": 1000, "v0_tolerance": 0, "max_iterations": 100},
    {"t": 100, "p": 2, "q": 3.1, "r": 1000, "v0_tolerance": 0, "max_iterations": 1},
]

KEYFRAMES[0] = {**DEFAULT_FRAME, **KEYFRAMES[0]}

TIMELINE = []
VARIABLE_TIMELINES = {}
IMAGE_RANGE = []
OUTPUT_FORMAT = '%04d.png'
OUTPUT_DIR = pathlib.Path()



def refl(vector, mir):
    return vector - 2*numpy.dot(vector,mir)*mir


def unit(vector):
    magnitude = sqrt(abs(numpy.dot(vector,vector)))
    return vector/magnitude


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def easeinout(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return (-2 * t * t) + (4 * t) - 1


def color_depth(rgb, depth, accuracy):

    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = h
    s = s*depth
    v = v*depth
    r,g,b = colorsys.hsv_to_rgb(h,s,v)
    a = depth*255*accuracy

    # hls = colorsys.rgb_to_hls(*rgb)
    # h = hls[0]
    # l = hls[1]*depth
    # s = hls[2]*depth
    # rgb_new = tuple(map(int, colorsys.hls_to_rgb(h,l,s)))
    # rgb_new = (*rgb_new, round(accuracy*255))

    # rgb_new = (*rgb, round(depth*255))

    return tuple(map(int, [r,g,b,a]))


def draw(pqr, max_iterations, vcolors, vtolerances):
    if not pqr:     # all pairs are asymptotic
        ir3 = 1/sqrt(3)
        mirror = [numpy.array([1j*ir3, 2*ir3, 0]),
                  numpy.array([1j*ir3, -ir3, -1]),
                  numpy.array([1j*ir3, -ir3,  1])]
    else:
        p = pqr.pop(0)
        pangle = pi/p
        cosqr = [ -cos(pi/u) for u in pqr ]
        while len(cosqr) < 2:
            cosqr.append(-1)

        v0 = [0,1,0]
        v11 = -cos(pangle)
        v12 = sin(pangle)
        v1 = [ 0, v11, v12 ]
        v21 = cosqr[0]
        v22 = (cosqr[1] - v11*v21) / v12
        v2 = [ 1j*sqrt(abs(1-v21**2-v22**2)), v21, v22 ]
        mirror = [ numpy.array(v0), numpy.array(v1), numpy.array(v2) ]

        # ## Move everything so that the origin is equidistant from the mirror.
        # omnipoint = unit(numpy.linalg.solve(numpy.array(mirror), numpy.array([-1,-1,-1])))
        # if omnipoint[0].imag < 0:
        #     omnipoint = -omnipoint
        # tempmirror = unit(omnipoint - numpy.array([1j,0,0]))
        # for j,u in enumerate(mirror):
        #     v = refl(u,tempmirror)
        #     if v[0].imag <0: v = -v
        #     mirror[j] = v

    v0,v1,v2 = mirror
    
    vertex = numpy.linalg.solve(numpy.array(mirror), numpy.array([0,1,1]))
    vertex = numpy.linalg.solve(numpy.array(mirror), numpy.array([1,0,1]))
    critplane = 1j*numpy.cross(vertex,v2)

    depth_map = (1 - (numpy.array(range(max_iterations+1)) / max_iterations)) ** 2

    
    def render_pixel(x, y):
        nonlocal max_iterations, depth_map

        r2 = x**2 + y**2
        if r2 >= 1:
            return TRANSPARENT

        bottom = 1-r2
        p = numpy.array([ 1j*(1+r2)/bottom, 2*x/bottom, 2*y/bottom ])

        clean = 0
        for iteration in range(max_iterations):
            for j,u in enumerate(mirror):
                # attempts += 1
                if numpy.dot(p,u) > 0:
                    p = refl(p,u)
                    clean = 0
                else:
                    clean += 1
                    if clean >= 3:
                        # if numpy.dot(p,critplane) > 1: return 1, attempts
                        # if attempts > max_attempts: max_attempts = attempts
                        
                        v0_tolerance = vtolerances[0]
                        v0_dot = abs(numpy.dot(p,v0))
                        if v0_dot < v0_tolerance:
                            depth = depth_map[iteration]
                            accuracy = ((v0_tolerance-v0_dot)/v0_tolerance)**(1/3)
                            return color_depth(vcolors[0], depth, accuracy)
                        # if abs(numpy.dot(p,v1)) < vtolerances[1]:  return vcolors[1]
                        # if abs(numpy.dot(p,v2)) < vtolerances[2]:  return vcolors[2]
                        return TRANSPARENT
        return TRANSPARENT
    

    im_data = [render_pixel(x,y) for y in IMAGE_RANGE for x in IMAGE_RANGE]
    im = Image.new("RGBA", (WIDTH_2, WIDTH_2) )
    im.putdata(im_data)
    return im


def render(frame, index):
    max_iterations = round(frame['max_iterations'])
    p = frame['p']
    q = frame['q']
    r = frame['r']
    pqr = list(filter(bool, [p,q,r]))
    
    v0_tolerance = frame.get('v0_tolerance')
    v1_tolerance = frame.get('v1_tolerance')
    v2_tolerance = frame.get('v2_tolerance')
    vtolerances = [v0_tolerance, v1_tolerance, v2_tolerance]

    v0_color = numpy.uint8(frame.get("v0_color"))
    v1_color = numpy.uint8(frame.get("v1_color"))
    v2_color = numpy.uint8(frame.get("v2_color"))
    vcolors = [v0_color, v1_color, v2_color]
    
    quad = draw(pqr, max_iterations, vcolors, vtolerances)
    image = Image.new("RGBA", (WIDTH, WIDTH))
    image.alpha_composite(quad, dest=(WIDTH_2,WIDTH_2))
    image.alpha_composite(quad.transpose(Image.Transpose.FLIP_LEFT_RIGHT), dest=(0,WIDTH_2))
    image.alpha_composite(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), dest=(0,0))
    background = Image.new("RGBA", (WIDTH, WIDTH), BACKGROUND_COLOR)
    background.alpha_composite(image)
    background.save(f"{OUTPUT_DIR}/{OUTPUT_FORMAT % index}")
    return index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', '-F',
                        type=int,
                        default=30,
                        help="Frames to render per second of DURATION.")
    parser.add_argument('--duration', '-D',
                        type=float,
                        default=4.0,
                        help="Total length in seconds of the resulting animation.")
    parser.add_argument('--width', '-W',
                        type=int,
                        default=100,
                        help="Width in pixels of the frames to be rendered.")
    parser.add_argument('--output-dir', '-d',
                        type=str,
                        default='.',
                        help="Directory in which to output rendered images. Defaults to $PWD.")
    parser.add_argument('--output-format', '-f',
                        type=str,
                        default='%04d.png',
                        help="""Formattable file-name for rendered images. The frame's index will
                                be supplied as a formatting argument to this string, e.g. a value
                                of '%%04d.png' would cause the first frame to be saved as 
                                '0001.png', etc. Defaults to %%04d.png""")
    parser.add_argument('--clear-output-dir', '-R',
                        action='store_true',
                        help="Delete all files in the output directory before generating frames.")
    parser.add_argument('--procs', '-P',
                        type=int,
                        default=4,
                        help="""Number of concurrent processes to launch. Each process renders a
                                single frame. It's recommended to keep this value less than or
                                equal to the number of cores your machine's CPU has.""")
    parser.add_argument('--ffmpeg',
                        nargs='?',
                        type=str,
                        const='render.mp4',
                        help="""Calls `ffmpeg` to encode the rendered frames as a video file using 
                                the same parameters given for the frame synthesis. If an argument
                                value is provided, it will be used as the resulting video file
                                name. This can be used specify the video format. Defaults to
                                'render.mp4'. \033[1mThe `ffmpeg` executable must be available in
                                $PATH.\033[0m""")
    args = parser.parse_args()

    FPS = args.fps
    DURATION = args.duration
    FRAMES = round(FPS * DURATION)
    WIDTH = args.width
    WIDTH_2 = WIDTH // 2
    # WIDTH_RANGE = numpy.linspace(-1.0, 1.0, WIDTH)
    IMAGE_RANGE = numpy.linspace(0, 1.0, WIDTH_2)
    TIMELINE = [{**KEYFRAMES[0]} for f in range(FRAMES)]


    OUTPUT_DIR = pathlib.Path(args.output_dir or '').resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.clear_output_dir:
        for i in OUTPUT_DIR.iterdir():
            i.unlink()

    OUTPUT_FORMAT = args.output_format

    print(f"Rendering {FRAMES} of {WIDTH}x{WIDTH} frames over {DURATION} seconds ({FPS}fps, {WIDTH*WIDTH*FRAMES} pixels) with"
          f" {args.procs} concurrent renderers\n  ==> {OUTPUT_DIR}/{OUTPUT_FORMAT}")


    for frame in KEYFRAMES:
        t = frame.pop("t")
        for k,v in frame.items():
            timeline = VARIABLE_TIMELINES.get(k,[])
            ramp = {"t":t, k:v}
            timeline.append(ramp)
            VARIABLE_TIMELINES[k] = timeline


    for variable, keyframes in VARIABLE_TIMELINES.items():
        ramp_iter = iter(VARIABLE_TIMELINES[variable])
        ramp = next(ramp_iter)


        while True:
            try: ramp_next = next(ramp_iter)
            except StopIteration: break
            t = ramp.pop("t")
            frame_start = math.floor(t * FRAMES / 100)
            frame_end = math.ceil(ramp_next['t'] * FRAMES / 100)
            frame_duration = frame_end - frame_start
            unit_duration = numpy.linspace(0, 1, frame_duration)
            ramp_over_frames = numpy.apply_along_axis(easeinout, arr=unit_duration[:,numpy.newaxis], axis=1).flatten()
            for k,v in ramp.items():
                v_start = v
                v_end = ramp_next[k]
                try:
                    v_delta = v_end - v_start
                    variable_ramp_over_frames = ramp_over_frames * v_delta + v_start
                except TypeError:
                    v_start = numpy.uint8(v_start)
                    v_end = numpy.uint8(v_end)
                    v_delta = v_end - v_start
                    variable_ramp_over_frames = ramp_over_frames[:,numpy.newaxis]*v_delta + v_start
                for f,f0 in zip(range(frame_start, frame_end), range(frame_duration)):
                    TIMELINE[f][k] = variable_ramp_over_frames[f0]
            ramp = ramp_next

    print(f"{0} / {FRAMES}", end='\r')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.procs) as pool:
        jobs = concurrent.futures.as_completed(pool.submit(render, frame, frame_index)
                                               for frame_index, frame in enumerate(TIMELINE))
        for future in jobs:
            index = future.result()
            print(f"{index+1} / {FRAMES}", end='\r')
            sys.stdout.flush()
    print("\033[2KComplete")


    if args.ffmpeg:
        status, ffmpeg_path = subprocess.getstatusoutput('command -v ffmpeg')
        if status != 0:
            raise FileNotFoundError(f"Could not find `ffmpeg` executable in $PATH.")
        ffmpeg_command = (f"{ffmpeg_path} -y"
                          f" -framerate {FPS} -r {FPS}"
                          f" -i {OUTPUT_DIR}/{OUTPUT_FORMAT}"
                          f" -- {OUTPUT_DIR}/{args.ffmpeg}")
        print(ffmpeg_command)
        print(subprocess.getoutput(ffmpeg_command))
