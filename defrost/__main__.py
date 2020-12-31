#!/usr/bin/env python2
# coding: utf-8
from __future__ import print_function, unicode_literals

import re
import os
import sys
import time
import json
import base64
import pprint
import codecs
import struct
import hashlib
import logging
import argparse
import platform
import threading
import traceback
import unicodedata
import subprocess as sp
from datetime import datetime

try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

"""
defrost.py: split broken icecast recordings into separate mp3s
2020-12-31, v0.9, ed <irc.rizon.net>, MIT-Licensed
https://ocv.me/dev/?defrost.py

status:
  kinda works

howto:
  this requiers an icecast recording which includes the
  icecast metadata; to create such a recording do this:
  
    wget -U 'MPlayer' --header "Icy-MetaData: 1" -S "https://stream.r-a-d.io/main.mp3"

new in this version:
  less sanity checks because FFmpeg cray
  support for read-only/nas sources

NOTE:
  the MP3s will be timestamped based on the source file, so
  if the original mp3 lastmodified was when recording ended
  then MP3s get lastmodified-times matching when they aired

NOTE:
  separate MP3s will be created for each track,
  if you prefer a single MP3 and a cuesheet then
  you can just concatenate the MP3s back together
  and create a cuesheet based on the standalone ones
"""

try:
    import mutagen
    import chardet
except ImportError as ex:
    print(
        "{0}\n\n  need {1}; please do this:\n    python -m pip install --user -U {1}".format(
            repr(ex), str(ex).split(" ")[-1].strip("'")
        )
    )
    sys.exit(1)


YOLO = False  # cfg: True disengages safety checks
METAINT = 16000  # cfg: should match the headers from the icecast server
IO_BUFSZ = 512 * 1024  # cfg: read/write buffer size, probably optimal

PY3 = sys.version_info[0] > 2
WINDOWS = platform.system() == "Windows"
PYPY = platform.python_implementation() == "PyPy"

if PY3 and not PYPY:
    uprint = print
else:
    UPRINT = codecs.getwriter("utf-8")

    def uprint(*args, **kwargs):
        stream = sys.stdout.buffer if PY3 else sys.stdout
        kwargs["file"] = UPRINT(stream)
        print(*list(args), **kwargs)
        stream.flush()


logger = logging.getLogger(__name__)
debug = logger.debug
info = logger.info
warn = logger.warning
error = logger.error


class LoggerFmt(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            c = "1;30"
        elif record.levelno == logging.INFO:
            c = ";32"
        elif record.levelno == logging.WARN:
            c = ";33"
        else:
            c = "1;31"

        ts = datetime.utcfromtimestamp(record.created)
        ts = ts.strftime("%H:%M:%S.%f")[:-3]

        msg = record.msg
        if record.args:
            msg = msg % record.args

        return "\033[0;36m{}\033[0{}m {}\033[0m".format(ts, c, msg)


def configure_logger(debug):
    lv = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=lv,
        format="\033[36m%(asctime)s.%(msecs)03d\033[0m %(message)s",
        datefmt="%H%M%S",
    )
    lh = logging.StreamHandler(sys.stderr)
    lh.setFormatter(LoggerFmt())
    logging.root.handlers = []  # kill other loggers
    logger.handlers = [lh]  # make ours fancy
    logger.setLevel(lv)


def reprint(*args, **kwargs):
    msg = args[0]
    msg = "\033[A{}\033[K".format(msg)
    uprint(msg, *list(args)[1:], **kwargs)


class IcyVerifier(object):
    """
    strict parser, assumes correct metaint, raises on corruption
    (see IcyScanner for the lenient version of this)
    """

    def __init__(self, fn):
        self.fn = fn
        self.find = b"StreamTitle='"
        self.icystep = METAINT + 1
        self.f = open(self.fn, "rb", IO_BUFSZ)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def run(self):
        while True:
            buf = self.f.read(self.icystep)
            if len(buf) < self.icystep:
                return

            sz = struct.unpack("B", buf[-1:])[0] * 16
            if sz == 0:
                continue

            ofs = self.f.tell()
            buf = self.f.read(sz).rstrip(b"\0")
            if not buf.startswith(self.find):
                raise Exception("bad magic @ {} / {:x}".format(ofs, ofs))

            yield [ofs - 1, buf]


class IcyScanner(object):
    """
    find metadata without relying on metaint
    (use this instead of IcyVerifier to determine metadata location)
    """

    def __init__(self, fn):
        self.fn = fn
        self.find = b"StreamTitle='"
        self.f = open(self.fn, "rb", IO_BUFSZ)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def run(self):
        bufsz = 256 * 16 + 1
        buf = b""
        pos = 0
        while True:
            b = self.f.read(bufsz)
            if not b:
                return

            buf += b
            while True:
                ofs = buf.find(self.find)
                if ofs < 0:
                    if len(buf) > bufsz:
                        buf = buf[bufsz:]
                        pos += bufsz
                    break

                if len(buf) - ofs <= 256 * 16:
                    break

                sz = struct.unpack("B", buf[ofs - 1 : ofs])[0] * 16
                mbuf = buf[ofs : ofs + sz]

                testbuf = mbuf.rstrip(b"\0")
                if not testbuf.endswith(b"';"):
                    raise Exception(
                        "bad suffix @ {0} / {0:x} (fullbuf {1} / mbuf {2} / testbuf {3})".format(
                            pos + ofs, len(buf), len(mbuf), len(testbuf)
                        )
                    )

                yield [pos + ofs - 1, mbuf]
                buf = buf[ofs + sz :]
                pos += ofs + sz


def zerochk(f, lower, upper, lower_is_origin):
    if lower_is_origin:
        # print("\033[1;30mzerochk from lower {:x} {:x}\033[0m".format(lower, upper))
        ofs = lower - 1
    else:
        # print("\033[1;30mzerochk from upper {:x} {:x}\033[0m".format(lower, upper))
        ofs = upper
        while True:
            ofs -= METAINT + 1
            if ofs < lower:
                break

    ret = []
    while True:
        ofs += METAINT + 1
        if ofs >= upper:
            break

        f.seek(ofs)
        buf = f.read(1)
        # print("try {0} {0:x} = {1}".format(ofs, int(buf[0])))
        if buf == b"\00":
            ret.append(ofs)
        else:
            if lower_is_origin:
                break
            else:
                ret = []

    return ret


def tag2text(buf):
    buf = buf.rstrip(b"\x00")
    err = ""
    try:
        enc = "utf-8"
        meta = buf.decode(enc)
    except:
        try:
            enc = chardet.detect(buf)["encoding"]
            meta = buf.decode(enc)
        except:
            enc = "utf-8"
            meta = buf.decode(enc, "ignore")
            err += " MOJIBAKE"

    safe_meta = "".join(c for c in meta if unicodedata.category(c)[0] != "C")

    if meta != safe_meta:
        meta = safe_meta
        err += " CONTROL"
        # uprint(u'[{}]\n[{}]'.format(meta,meta2))

    return [meta, enc, err]


def fmt_meta(metasrc_yield, sz, ntags):
    pos, buf = metasrc_yield
    meta, enc, err = tag2text(buf)

    return "at {}/{} ({:.2f}%), {} tags, [{}] [{}] {}".format(
        pos, sz, (pos * 100.0 / sz), ntags, enc, meta, err
    )


def ensure_parsers_agree(metasrc1, metasrc2, sz):
    ntags = 0
    for md1, md2 in zip_longest(metasrc1.run(), metasrc2.run(), fillvalue=None):
        ntags += 1
        pos1, buf1 = md1 or [0, "EOF"]
        pos2, buf2 = md2 or [0, "EOF"]
        err = pos1 != pos2 or buf1 != buf2
        if err or ntags % 20 == 0:
            reprint(fmt_meta(md1, sz, ntags))

        if err:
            msg = "scanner/verifier disagreement,\n{} [{}]\n{} [{}]".format(
                pos1,
                tag2text(buf1)[0],
                pos2,
                tag2text(buf2)[0],
            )
            msg += "\nthis is not an error (should just be ignored) "
            msg += "but raising for inspection until properly tested"
            raise Exception(msg)


def defrost(fn, metasrc, sz):
    """
    yields [lower, upper, zeros, tags]
    """
    yield_ofs = 0
    zeros = []
    tags = []
    ofs1 = 0
    bm1 = None
    f = open(fn, "rb", IO_BUFSZ)
    # iterate over all detected tags,
    # consider ranges between previous and current tag
    for ofs2, bm2 in metasrc.run():
        tags.append(bm2)
        # if len(tags) % 10 == 0:
        #     # uprint("{0} {0:x} {1}{2}".format(ofs2, *tag2text(bm2)))
        #     reprint(fmt_meta([ofs2, bm2], sz, len(tags)))

        if not bm1:
            if ofs2 != METAINT:
                raise Exception(
                    "initial tag at {:x}, expected at {:x}".format(ofs2, METAINT)
                )
            after_tag = ofs1
            zeros_up = []
        else:
            after_tag = ofs1 + len(bm1) + 1
            zeros_up = zerochk(f, after_tag, ofs2, True)

        zeros_dn = zerochk(f, after_tag, ofs2, False)

        # default to yielding all data between previous and current tag
        mp3_ofs1 = after_tag
        mp3_ofs2 = ofs2

        if bm1 and ofs2 - ofs1 == METAINT + len(bm1) + 1:
            # immediate tag change with correct metaint spacing
            zeros_up = []
            zeros_dn = []

        elif zeros_up and zeros_up == zeros_dn:
            # another best case scenario; no corruption
            zeros.extend(zeros_up)
            zeros_up = []
            zeros_dn = []

        else:
            # something's fucky
            if bm1 and not zeros_up and not zeros_dn:
                raise Exception(
                    "all hope is lost between {:x} and {:x}, bm1 {:x}, bm2 {:x}".format(
                        ofs1, ofs2, len(bm1), len(bm2)
                    )
                )

            gap_lower = ofs1
            if zeros_up:
                gap_lower = zeros_up[-1]
                print(
                    "\033[35mUP {}\n".format(
                        " ".join("{:x}".format(x) for x in zeros_up)
                    )
                )

            gap_upper = ofs2
            if zeros_dn:
                gap_upper = zeros_dn[0]
                print(
                    "\033[33mDN {}\n".format(
                        " ".join("{:x}".format(x) for x in zeros_dn)
                    )
                )

            print(
                "\033[0;1;34mofs1 {:x}   ofs2 {:x}   g_l {:x}   g_u {:x}   steps {:.2f}\033[0m\n".format(
                    ofs1,
                    ofs2,
                    gap_lower,
                    gap_upper,
                    (gap_upper - gap_lower) * 1.0 / METAINT,
                )
            )

            if gap_lower > gap_upper:
                # we could parse the mp3 headers to figure this one out
                # but let's just return the same data twice #yolo
                raise Exception(
                    "corruption between {:x} and {:x} causes overlapping recovery; {:x} from start vs {:x} from end".format(
                        ofs1, ofs2, gap_lower, gap_upper
                    )
                )

            if gap_upper - gap_lower > METAINT + 1:
                # best sync guess leaves a chunk of unknown between the tags,
                # include it with the previous track and hope ffmpeg fixes it
                gap_lower = gap_upper - METAINT

                # raise Exception(
                #     "multiple desyncs between {:x} and {:x}, gap from {:x} to {:x} (length {:x})".format(
                #         ofs1, ofs2, gap_lower, gap_upper, gap_upper - gap_lower
                #     )
                # )

            if zeros_up:
                zeros.extend(zeros_up)

            mp3_ofs1 = yield_ofs
            mp3_ofs2 = gap_lower + METAINT

        yield [mp3_ofs1, mp3_ofs2, zeros, tags]
        yield_ofs = mp3_ofs2
        tags = tags[-1:]
        zeros = zeros_dn
        ofs1 = ofs2
        bm1 = bm2

    # final readout
    fsz = os.path.getsize(fn)
    after_tag = ofs1 + len(bm1) + 1
    # zeros.extend([ofs2, after_tag])
    zeros.extend(zerochk(f, after_tag, fsz, True))
    yield [after_tag, fsz, zeros, tags]


def run_defrost(fn, fn_mp3, fn_idx, sz):
    with open(fn, "rb", IO_BUFSZ) as fi, open(fn_mp3, "wb", IO_BUFSZ) as fom, open(
        fn_idx, "w", IO_BUFSZ
    ) as foi, IcyScanner(fn) as metasrc:
        ntags = 0
        fom_pos = 0
        for lower, upper, zeros, tags in defrost(fn, metasrc, sz):
            tags = [x.rstrip(b"\x00")[13:-2] for x in tags]
            ntags += 1
            if ntags % 10 == 0 or True:
                perc = lower * 100.0 / sz
                msg = "{0:.2f}%  #{1}  {2:x}..{3:x}  {2:}..{3:}  z{4}  t{5}  [{6}]".format(
                    perc,
                    ntags,
                    lower,
                    upper,
                    len(zeros),
                    len(tags),
                    tag2text(tags[0])[0],
                )
                reprint(msg)

            inf = {
                "o": fom.tell() or fom_pos,
                "o1": lower,
                "o2": upper,
                "z": [x - lower for x in zeros],
                "t": base64.urlsafe_b64encode(tags[0]).decode("utf-8"),
            }
            jtxt = json.dumps(inf, sort_keys=True)
            foi.write(jtxt + "\n")
            if False:
                # just idx, skip mp3
                fom_pos += upper - lower - len(zeros)
                continue

            fi.seek(lower)
            ptr = lower
            rem = upper - lower
            while rem > 0:
                buf = fi.read(min(IO_BUFSZ, rem))
                bsz = len(buf)

                # remove metaint zerobytes from buffer
                buf2 = b""
                b2ofs = 0
                zlist = [x - ptr for x in zeros if x >= ptr and x < ptr + bsz]
                zlist += [bsz]
                for zofs in zlist:
                    # print("in range {} to {}, drop {}".format(ptr, ptr + bsz, zofs))
                    buf2 += buf[b2ofs:zofs]
                    b2ofs = zofs + 1
                buf = buf2
                # print()

                ptr += bsz
                rem -= bsz
                fom.write(buf)
                continue

                while buf:
                    p1 = fom.tell()
                    n = fom.write(buf)
                    p2 = fom.tell()
                    n = n or p2 - p1
                    buf = buf[n:]
                    if n == 0:
                        raise Exception("writes to mp3 file are being rejected")
                    elif buf:
                        msg = "truncated write, {}-{}={}".format(bsz, n, bsz - n)
                        raise Exception(msg)


def collect_frames(fn_mp3):
    # cmd /c "set FFREPORT=level=32:file=ffprobe.log & ffprobe -hide_banner -select_streams a:0 -show_frames -show_entries frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pkt_duration_time,pkt_pos,nb_samples -of compact=p=0 5g-ok.mp3.defrost.mp3 >ffprobe.out"
    # tail -F ffprobe.out | awk 'NR%1000==1'
    # perl -e 'while(<>){if (!/^pkt_pts_time=([0-9\.]+)\|pkt_dts_time=([0-9\.]+)\|best_effort_timestamp_time=([0-9\.]+)\|pkt_duration_time=([0-9\.]+)\|pkt_pos=([0-9]+)\|nb_samples=([0-9]+)\r?$/) {print$_;next} printf "%.3f %.3f %.3f %.3f %d %d\n",$1,$2,$3,$4,$5,$6}' <ffprobe.out >ffprobe.outf

    ptn = r"^pkt_pts_time=([0-9\.]+)\|pkt_dts_time=([0-9\.]+)\|best_effort_timestamp_time=([0-9\.]+)\|pkt_pos=([0-9]+)\r?$"
    # ptn = r"^pts_time=([0-9\.]+)\|dts_time=([0-9\.]+)\|pos=([0-9]+)\r?$"
    ptn = re.compile(ptn)

    # fmt: off
    c = [
        "ffprobe",
        "-hide_banner",
        "-select_streams", "a:0",
        "-show_frames",
        # "-show_packets",
        "-show_entries", "frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pkt_pos",
        # "-show_entries", "packet=pts_time,dts_time,pos",
        "-of", "compact=p=0",
        fn_mp3,
    ]
    # fmt: on

    # show_frames: 243.94 sec
    # show_packets: 61.10 sec (but includes some additional frames, dangeruss)

    p = sp.Popen(c, stdout=sp.PIPE)
    fails = 0
    buf = b""
    while True:
        ibuf = p.stdout.read(4096)
        if not ibuf:
            if p.poll() is not None:
                info("FFmpeg terminated")
                return
            else:
                fails += 1
                if fails < 30:
                    time.sleep(0.1)
                    continue

                raise Exception("read err 1")

        fails = 0
        buf += ibuf
        try:
            tbuf, buf = buf.rsplit(b"\n", 1)
        except:
            continue

        sbuf = tbuf.decode("utf-8")
        for ln in sbuf.split("\n"):
            m = ptn.match(ln)
            if not m:
                continue

            pts, dts, ts, pos = m.groups()
            if pts != dts or pts != ts:
                warn("defrost: spooky timestamps ({},{},{})\n".format(pts, dts, ts))

            # ts, dts, pos = m.groups()
            # if ts != dts:
            #     warn("defrost: spooky timestamps ({},{})\n".format(ts, dts))

            yield [int(pos), float(ts)]


def detect_silence_one(fn_mp3, db, len_sec, xpos, ret):
    """
    yields [nFrame, dB, startTime, endTime]
    """

    # ffmpeg -i 5g-ok.mp3.defrost.mp3 -af silencedetect=n=-30dB:d=1,silencedetect=n=-40dB:d=1,silencedetect=n=-50dB:d=1,silencedetect=n=-60dB:d=1 -f null -
    # cmd /c "set FFREPORT=level=32:file=ffsilence.log & ffmpeg -v fatal -i 5g-ok.mp3.defrost.mp3 -af silencedetect=n=-30dB:d=1,ametadata=print:file=silence.txt -f null -"
    # awk '{sub(/.*\r/,"");m=0} /monotonic/{m=1} m&&pm{next} {pm=m;m=0} 1' < ffsilence.log
    # ffmpeg -i 5g-ok.mp3.defrost.mp3 -af silencedetect=n=-30dB:d=1,silencedetect=n=-40dB:d=1,silencedetect=n=-50dB:d=1,silencedetect=n=-60dB:d=1,ametadata=mode=print -f null -
    # ffmpeg -i 5g-ok.mp3.defrost.mp3 -af ametadata=mode=delete,silencedetect=n=-30dB:d=0.5,ametadata=mode=print,ametadata=mode=delete,silencedetect=n=-40dB:d=0.5,ametadata=mode=print,ametadata=mode=delete,silencedetect=n=-50dB:d=0.5,ametadata=mode=print,ametadata=mode=delete,silencedetect=n=-60dB:d=0.5,ametadata=mode=print -f null -

    ptn1 = re.compile(r"^frame:([0-9]+) +pts:([0-9]+) +pts_time:([0-9\.]+)$")
    ptn2 = re.compile(r"^lavfi\.silence_duration=([0-9\.]+)$")

    # fmt: off
    c = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v", "fatal",
        "-i", fn_mp3,
        "-af", "ametadata=mode=add:key=a:value=a,silencedetect=n=-{}dB:d=0.7,ametadata=mode=print:file=-".format(db),
        "-f", "null", "-",
    ]
    # fmt: on

    bsz = 1024 * 64
    p = sp.Popen(c, stdout=sp.PIPE, bufsize=bsz)
    # logfile = open("defrost-{}db-{}.log".format(db, os.getpid()), "wb")
    # logfile_en = False
    buf = b""
    pframe = 0
    nframe = -1
    frame_ts = -1
    unroll = 0
    last_print = 0
    while True:
        fails = 0
        ibuf = p.stdout.read(bsz)
        if not ibuf:
            if p.poll() is not None:
                # info("FFmpeg terminated")
                return
            else:
                fails += 1
                if fails < 30:
                    time.sleep(0.02)
                    continue

                raise Exception("read err 1")

        # if b"\nframe:3042000 " in ibuf:
        #     logfile_en = True
        # if logfile_en:
        #     logfile.write("\n[ {} ]\n".format(len(ibuf)).encode("utf-8"))
        #     logfile.write(ibuf)

        buf += ibuf.replace(b"\r", b"\n")
        try:
            tbuf, buf = buf.rsplit(b"\n", 1)
        except:
            continue

        sbuf = tbuf.decode("utf-8")
        for ln in sbuf.split("\n"):
            m = ptn1.match(ln)
            if m:
                g = m.groups()
                nframe, pts = [int(x) for x in g[:2]]
                frame_ts = float(g[2])
                # ffmpeg drops precision to int after 100k sec, fix it
                if frame_ts > 9000:
                    frame_ts = pts / (1.0 * round(pts / frame_ts))
                # ffmpeg randomly starts counting from frame 0 again, fix it
                if nframe < pframe / 2:
                    delta = pframe - nframe
                    msg = "unroll {}+{}={}, {}dB ({}<{}), {}sec\n".format(
                        unroll, delta, unroll + delta, db, nframe, pframe, frame_ts
                    )
                    warn(msg)
                    unroll += delta
                pframe = nframe
                nframe += unroll
                continue

            m = ptn2.match(ln)
            if not m:
                continue

            dur = float(m.group(1))
            t0 = frame_ts - dur
            if ret:
                prev_frame, _, prev_t0, prev_ts = ret[-1]
                if nframe < prev_frame or t0 < prev_t0 or frame_ts < prev_ts:
                    msg = "rollback in {}dB:\n  {}  {}  {}\n  {}  {}  {}\n".format(
                        db, prev_frame, prev_t0, prev_ts, nframe, t0, frame_ts
                    )
                    error(msg)
                    raise Exception(msg)

            ret.append([nframe, db, t0, frame_ts])

        now = time.time()
        if now - last_print > 1:
            last_print = now
            msg = "\033[A {3:.1f}% \033[{0}G{1}x {2}dB, {3:.1f}% \n".format(
                xpos + 12, len(ret), db, frame_ts * 100.0 / len_sec
            )
            print(msg, end="")


def build_framecache(f_frames, ofs1, ofs2):
    lower = ofs1 - 192 * 128 * 300
    upper = ofs2 + 192 * 128 * 3600
    msg = "rebuild framecache {}..{} for {}..{}".format(lower, upper, ofs1, ofs2)
    debug(msg)
    nframe = 0
    framecache = []

    f_frames.seek(0, os.SEEK_END)
    txt_sz = f_frames.tell()
    hop_sz = hop_pos = txt_sz / 2
    while hop_sz > 9000:
        hop_sz /= 2
        f_frames.seek(hop_pos)
        next(f_frames)
        ofs = int(next(f_frames).split(" ")[1])
        if ofs > ofs1 - 9000:
            hop_pos = max(0, hop_pos - hop_sz)
        else:
            hop_pos += hop_sz

    for ln in f_frames:
        nframe, ofs, sec = ln.split(" ")
        nframe = int(nframe)
        ofs = int(ofs)
        sec = float(sec)
        if ofs < lower:
            continue
        if ofs > upper:
            break
        framecache.append([nframe, ofs, sec])

    if not framecache:
        msg = "could not build framecache ofs {}..{}, last seen ofs {}, sec {}"
        raise Exception(msg.format(lower, upper, ofs, sec))

    return framecache


def find_silence(silents, target_sec):
    best_sec = None
    best_frame = None
    best_dist = None
    best_db = None
    for db in [60, 50, 40, 30]:
        if db <= 30 and best_dist:
            # skip 30dB if we have anything at all
            break

        for nframe, t1, t2 in silents[db]:
            if t2 < target_sec - 30:
                continue

            if t1 > target_sec + 30:
                break

            # NOTE keep the offsets above the same
            #   maybe it could go skipping some segments otherwise
            #   (too tired to actually check if thats the case)

            t = t1 + (t2 - t1) * 0.8
            dist = abs(target_sec - t)
            if not best_dist or dist < best_dist:
                if not best_db or dist < best_dist / 4:
                    best_dist = dist
                    best_frame = nframe
                    best_sec = t

    if best_sec:
        return [best_sec, best_frame]
    else:
        return [target_sec, None]


def sanitize_fn(fn):
    for bad, good in [
        ["<", "＜"],
        [">", "＞"],
        [":", "："],
        ['"', "＂"],
        ["/", "／"],
        ["\\", "＼"],
        ["|", "｜"],
        ["?", "？"],
        ["*", "＊"],
        ["'", "＇"],  # shell-safety
        ["`", "｀"],  # shell-safety
    ]:
        fn = fn.replace(bad, good)

    if WINDOWS:
        bad = ["con", "prn", "aux", "nul"]
        for n in range(1, 10):
            bad += "com{0} lpt{0}".format(n).split(" ")

        if fn.lower() in bad:
            fn = "_" + fn

    return fn


def split_mp3(f_defrosted, ntrack, ofs1, ofs2, outdir, tagtxt):
    title = sanitize_fn(tagtxt) or "untitled"
    while True:
        try:
            fn = "{}/{}. {}.mp3".format(outdir, ntrack, title)
            f = open(fn, "wb")
            break
        except:
            if not title:
                raise
            title = title[:-1]

    f_defrosted.seek(ofs1)
    rem = ofs2 - ofs1
    while rem > 0:
        buf = f_defrosted.read(4096)
        rem -= len(buf)
        if rem and not buf:
            raise Exception("hit eof in defrosted mp3")

        f.write(buf)

    f.close()

    artist = None
    ofs = tagtxt.find(" - ")
    if ofs < 0:
        title = tagtxt
    if ofs == 0:
        title = tagtxt[3:]
    elif ofs > 0:
        artist = tagtxt[:ofs]
        title = tagtxt[ofs + 3 :]

    try:
        debug("writing tags")
        mt = mutagen.File(fn, easy=True)
        mt.add_tags()

        mt["title"] = title
        if artist:
            mt["artist"] = artist

        mt.save(v2_version=4)
    except Exception as ex:
        warn("failed to write tags: {}".format(repr(ex)))

    return fn


def stamp_mp3(fn, start_ts, eof_ts, lastmod):
    ts = lastmod - (eof_ts - start_ts)
    os.utime(fn, (int(time.time()), ts))
    return ts


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", action="store_true", help="enable debug output")
    ap.add_argument("-o", metavar="DIR", help="output directory")
    ap.add_argument("--no-split", action="store_true", help="do not split the mp3")
    ap.add_argument("ICY_MP3", help="icecast recording with inband metadata")
    ar = ap.parse_args()

    if WINDOWS:
        os.system("rem")  # best girl

    configure_logger(ar.d)

    rc = 0
    t0 = time.time()
    fn = ar.ICY_MP3
    sz = os.path.getsize(fn)
    lastmod = os.path.getmtime(fn)
    print("using {} ({:.2f} MiB)".format(fn, sz / 1024.0 / 1024))

    src_hash = hashlib.sha1(fn.encode('utf-8', 'ignore')).digest()
    src_hash = base64.urlsafe_b64encode(src_hash)[:8]
    base = fn.replace("\\", "/").split("/")[-1]
    localbase = base + ".defrost-" + src_hash
    
    fn_mp3 = localbase + ".mp3"
    fn_idx = localbase + ".idx"
    fn_frames = localbase + ".frames"
    fn_silence = localbase + ".silence"

    outdir = ar.o or "defrost-{}-{}-{}".format(base, src_hash, os.getpid())
    if not ar.no_split:
        os.mkdir(outdir)

    if not os.path.exists(fn_mp3):
        print("\n")
        info("running icy verification pass\n")
        try:
            with IcyVerifier(fn) as metasrc1:
                with IcyVerifier(fn) as metasrc2:
                    ensure_parsers_agree(metasrc1, metasrc2, sz)
        except:
            rc += 1
            exc = traceback.format_exc()
            uprint(exc + "\nWARNING: verification failed (will continue in 2sec)")
            time.sleep(2)

        print("\n")
        info("running decode pass")
        run_defrost(fn, fn_mp3, fn_idx, sz)

    sz = os.path.getsize(fn_mp3)
    if not os.path.exists(fn_frames):
        print("\n")
        info("collecting frames with FFmpeg")
        buf = []
        pos = 0
        nframes = 0
        with open(fn_frames, "w") as f:
            for pos, ts in collect_frames(fn_mp3):
                nframes += 1
                buf.append("{} {} {:.3f}\n".format(nframes, pos, ts))
                if len(buf) > 1024 * 16:
                    print("\033[A{:.2f}% ".format(pos * 100.0 / sz))
                    f.write("".join(buf))
                    buf = []
            if buf:
                f.write("".join(buf))

        if sz > pos + 4 * 1024:
            rc += 1
            msg = "\nWARNING:\n  something funky with the file,\n  {} bytes total but\n  {} bytes last-frame\n"
            warn(msg.format(sz, pos))

    print("\n")
    info("reading eof from frametab")
    with open(fn_frames, "r", IO_BUFSZ) as f:
        f.seek(0, os.SEEK_END)
        f.seek(f.tell() - 1024)
        next(f)
        for ln in f:
            end_frame, end_pos, end_ts = ln.split(" ")
            end_frame = int(end_frame)
            end_pos = int(end_pos)
            end_ts = float(end_ts)

    msg = "frame: {}\npos: {}\nts: {:.3f}"
    print(msg.format(end_frame, end_pos, end_ts))

    if not os.path.exists(fn_silence):
        print("\n")
        info("detecting silence with FFmpeg\n")
        tasks = []
        for n, db in enumerate([30, 40, 50, 60]):
            r = []
            args = (fn_mp3, db, end_ts, n * 22, r)
            t = threading.Thread(target=detect_silence_one, args=args)
            tasks.append([t, r])
            t.start()

        r = []
        for task in tasks:
            # task[0].join()
            while task[0].is_alive():
                time.sleep(0.1)

            r.extend(task[1])

        r.sort()
        with open(fn_silence, "w") as f:
            for nframe, db, start, end in r:
                ln = "{} {} {:.2f} {:.2f}\n".format(nframe, db, start, end)
                f.write(ln)

    # probably small enough
    print()
    info("reading silent ranges into memory")
    silents = {}  # dB => [nframe, t1, t2]  (prefer timestamps)
    with open(fn_silence, "r") as f:
        for ln in f:
            fields = ln.split(" ")
            nframe, db = [int(x) for x in fields[:2]]
            t1, t2 = [float(x) for x in fields[2:]]
            entry = [nframe, t1, t2]
            try:
                silents[db].append(entry)
            except:
                silents[db] = [entry]

    ntrack = 0
    framecache = []  # nframe, ofs, sec
    with open(fn_idx, "r") as f_idx, open(fn_frames, "r") as f_frames, open(
        fn_mp3, "rb"
    ) as f_defrosted:
        tag = None
        for ln in f_idx:
            tag2 = json.loads(ln)
            tag2 = {
                "o": tag2["o"],
                "t": base64.urlsafe_b64decode(tag2["t"].encode("ascii")),
            }
            if not tag:
                tag = tag2
                continue

            ntrack += 1
            ofs1 = tag["o"]
            ofs2 = tag2["o"]
            debug("track #{}, ofs {}..{}".format(ntrack, ofs1, ofs2))

            lower = ofs1 - 30
            upper = ofs2 + 30
            if not framecache or framecache[0][1] > lower or framecache[-1][1] < upper:
                if framecache:
                    msg = "framecache insufficient, {} > {} or {} < {}"
                    debug(msg.format(framecache[0][1], lower, framecache[-1][1], upper))
                framecache = build_framecache(f_frames, ofs1, ofs2)
                msg = "framecache covers frame {}..{}, ofs {}..{}, ts {}..{}"
                fc0 = framecache[0]
                fc1 = framecache[1]
                debug(msg.format(fc0[0], fc1[0], fc0[1], fc1[1], fc0[2], fc1[2]))

            tag_ts1 = None
            tag_ts2 = None
            frame1 = None
            frame2 = None
            for nframe, ofs, ts in framecache:
                # print("cmp {} {}".format(ofs, ofs2))
                if not tag_ts1 or ofs <= ofs1:
                    frame1 = nframe
                    tag_ts1 = ts
                elif not tag_ts2 or ofs <= ofs2:
                    frame2 = nframe
                    tag_ts2 = ts
                else:
                    break

            msg = "track #{}, ofs {}..{}, sec {}..{}, frame {}..{}"
            debug(msg.format(ntrack, ofs1, ofs2, tag_ts1, tag_ts2, frame1, frame2))

            ts1 = tag_ts1
            ts2 = tag_ts2
            quiet_ts1, quiet_frame1 = find_silence(silents, tag_ts1)
            quiet_ts2, quiet_frame2 = find_silence(silents, tag_ts2)
            for nframe, ofs, ts in framecache:
                if ts <= quiet_ts1 and quiet_frame1:
                    frame1 = nframe
                    ofs1 = ofs
                    ts1 = ts
                elif ts <= quiet_ts2 and quiet_frame2:
                    frame2 = nframe
                    ofs2 = ofs
                    ts2 = ts
                else:
                    break

            # ffmpeg does some sick drifts occasionally
            # so lets be extremely lenient with ffmpeg/ffprobe differences
            panik = False
            checks = [
                ["quiet1", quiet_frame1, quiet_ts1],
                ["quiet2", quiet_frame2, quiet_ts2],
                ["fc1", frame1, ts1],
                ["fc2", frame2, ts2]
            ]
            results = []
            for label, frame, ts in checks:
                if not frame or frame < 10 or ts < 1:
                    continue
                results.append(frame / ts)
            
            results.sort()
            if results and results[-1] / results[0] >= 1.05:  # bump if necessary, just guessing
                msg = "something desynced:\n{}".format(pprint.pformat([checks, results]))
                if YOLO:
                    error(msg)
                else:
                    raise Exception(msg)

            msg = "track #{}, ofs {}..{}, sec {}..{}, cut {}..{}, delta {:.2f}..{:.2f}"
            msg = msg.format(
                ntrack,
                ofs1,
                ofs2,
                tag_ts1,
                tag_ts2,
                ts1,
                ts2,
                abs(ts1 - tag_ts1),
                abs(ts2 - tag_ts2),
            )
            info(msg)

            tagbin = tag["t"]
            tagtxt, enc, _ = tag2text(tagbin)
            debug("tag: [{}] [{}]".format(enc, tagtxt))

            if not ar.no_split:
                fn = split_mp3(f_defrosted, ntrack, ofs1, ofs2, outdir, tagtxt)
                ts = stamp_mp3(fn, ts1, end_ts, lastmod)
                info("wrote [{}] [{}]".format(ts, fn))

            tag = tag2

    # print(tag2)
    print()
    fun = error if rc else info
    fun("finished in {:.2f} sec with {} errors".format(time.time() - t0, rc or "no"))
    sys.exit(rc)


if __name__ == "__main__":
    main()


# c:\Users\ed\bin\pypy3\pypy3.exe defrost.py 5g-ok.mp3
# c:\Users\ed\bin\pypy2\pypy.exe defrost.py 5g-ok.mp3
# defrost.py 5g-ok.mp3
# c:\Python27\python.exe defrost.py 5g-ok.mp3
#
# win32 pypy3: 14.57 sec
# win32 pypy2: 25.80 sec
# win64  cpy3: 20.18 sec
# win64  cpy2: 20.57 sec
# linux pypy3: 65.55 sec
# linux pypy2: 14.78 sec
# linux  cpy3:  8.35 sec
# linux  cpy2: 28.87 sec
