#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function, unicode_literals

about = {
    "name": "defrostir",
    "version": "0.15",
    "date": "2021-10-28",
    "description": "split broken icecast recordings into separate mp3s",
    "author": "ed",
    "license": "MIT",
    "url": "https://github.com/9001/defrost",
}

import re
import os
import sys
import time
import json
import base64
import pprint
import codecs
import shutil
import struct
import hashlib
import logging
import argparse
import calendar
import platform
import threading
import traceback
import unicodedata
import subprocess as sp

try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

"""
status:
  works pretty well

howto:
  this requiers an icecast recording which includes the
  icecast metadata; to create such a recording do this:

    wget -U 'MPlayer' --header "Icy-MetaData: 1" -S "https://stream.r-a-d.io/main.mp3"

  take note of the headers that the server sends back,
  especially the "icy-metaint" which should be 16000,
  otherwise you have to provide --metaint $yourvalue

additional supported input formats:
  - loopstream recording and its tags.txt
     (doesn't set the correct timestamps)

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

PY3 = sys.version_info[0] > 2
WINDOWS = platform.system() == "Windows"
PYPY = platform.python_implementation() == "PyPy"

miss = []
try:
    import mutagen
except ImportError:
    miss.append("mutagen")

try:
    import charset_normalizer as chardet
except ImportError:
    try:
        import chardet
    except ImportError:
        miss.append("charset_normalizer" if PY3 else "chardet")

if miss:
    exe = sys.executable
    m = "\n  please do this:\n    {0} -m pip install --user -U {1}"
    print(m.format(exe, " ".join(miss)))
    sys.exit(1)

from mutagen.id3 import ID3, TIT2, TPE1, TALB, COMM, TDRC


YOLO = False  # cfg: True disengages safety checks
METAINT = 16000  # cfg: should match the headers from the icecast server
IO_BUFSZ = 512 * 1024  # cfg: read/write buffer size, probably optimal


try:
    from datetime import datetime, timezone

    UTC = timezone.utc
except ImportError:
    from datetime import datetime, timedelta, tzinfo

    TD_ZERO = timedelta(0)

    class _UTC(tzinfo):
        def utcoffset(self, dt):
            return TD_ZERO

        def tzname(self, dt):
            return "UTC"

        def dst(self, dt):
            return TD_ZERO

    UTC = _UTC()


if not PY3:
    import io

    open = io.open


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

        ts = datetime.fromtimestamp(record.created, UTC)
        ts = ts.strftime("%H:%M:%S.%f")[:-3]

        msg = record.msg
        if record.args:
            msg = msg % record.args

        return "\033[0;36m%s\033[0%sm %s\033[0m" % (ts, c, msg)


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
    msg = "\033[A%s\033[K" % (msg,)
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

    return "at %s/%s (%.2f%%), %s tags, [%s] [%s] %s" % (
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
                pos1, tag2text(buf1)[0], pos2, tag2text(buf2)[0],
            )
            msg += "\nthis is not an error (should just be ignored) "
            msg += "but raising for inspection until properly tested"
            raise Exception(msg)


def defrost(fn, metasrc, sz):
    """
    yields [lower, upper, zeros, tags]
    """
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
                msg = "initial tag at {:x}, expected at {:x}".format(ofs2, METAINT)
                if YOLO:
                    error(msg)
                else:
                    raise Exception(msg)
            after_tag = ofs1
            zeros_up = []
        else:
            after_tag = ofs1 + len(bm1) + 1
            zeros_up = zerochk(f, after_tag, ofs2, True)

        zeros_dn = zerochk(f, after_tag, ofs2, False)

        if bm1 and ofs2 - ofs1 == METAINT + len(bm1) + 1:
            # immediate tag change with correct metaint spacing
            yield [after_tag, ofs2, [], tags]

        elif zeros_up and zeros_up == zeros_dn:
            # another best case scenario; no corruption
            yield [after_tag, ofs2, zeros_up, tags]

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

            elif gap_upper - gap_lower > METAINT + 1:
                # best sync guess leaves a chunk of unknown between the tags,
                # include it with the previous track and hope ffmpeg fixes it
                gap_lower = gap_upper - METAINT

                # raise Exception(
                #     "multiple desyncs between {:x} and {:x}, gap from {:x} to {:x} (length {:x})".format(
                #         ofs1, ofs2, gap_lower, gap_upper, gap_upper - gap_lower
                #     )
                # )

            else:
                # probably good data around the gaps; include it
                gap_lower = min(ofs2, gap_lower + METAINT - 1)
                gap_upper = max(ofs1, 1 + gap_upper - METAINT)

            if zeros_up:
                yield [after_tag, gap_lower, zeros_up, tags]

            if zeros_dn:
                yield [gap_upper, ofs2, zeros_dn, tags]

        tags = tags[-1:]
        ofs1 = ofs2
        bm1 = bm2

    # final readout
    fsz = os.path.getsize(fn)
    after_tag = ofs1 + len(bm1) + 1
    zeros = zerochk(f, after_tag, fsz, True)
    yield [after_tag, fsz, zeros, tags]


def run_defrost(fn, fn_mp3, fn_idx, sz):
    with open(fn, "rb", IO_BUFSZ) as fi, open(fn_mp3, "wb", IO_BUFSZ) as fom, open(
        fn_idx, "w", IO_BUFSZ, encoding="utf-8"
    ) as foi, IcyScanner(fn) as metasrc:
        ntags = 0
        fom_pos = 0
        for lower, upper, zeros, tags in defrost(fn, metasrc, sz):
            tags = [x.rstrip(b"\x00")[13:-2] for x in tags]
            ntags += 1
            if ntags % 5 == 0:
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
    # fmt: off
    c = [
        "ffprobe",
        "-hide_banner",
        "-select_streams", "a:0",
        "-show_frames",
        # "-show_packets",
        "-show_entries", "frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pkt_pos,channels",
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
            try:
                fields = [x.split("=") for x in ln.split("|")]
                fields = {k: v for k, v in fields}
                _ = fields["best_effort_timestamp_time"]
            except:
                continue

            pts = fields.get("pkt_pts_time")
            dts = fields.get("pkt_dts_time")
            ts = fields.get("best_effort_timestamp_time")
            if (
                (pts and dts and pts != dts)
                or (pts and ts and pts != ts)
                or (dts and ts and dts != ts)
            ):
                warn("defrost: spooky timestamps ({},{},{})\n".format(pts, dts, ts))

            ts = ts or dts or pts

            yield [int(fields["pkt_pos"]), float(ts)]


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
        "-af", "ametadata=mode=add:key=a:value=a,silencedetect=n=-{}dB:d=0.2,ametadata=mode=print:file=-".format(db),
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
    lower_byte = max(0, ofs1 - 320 * 128 * 60)
    extra_sec = 3600
    debug("rebuild framecache %s..+%s for %s..%s", lower_byte, extra_sec, ofs1, ofs2)
    nframe = 0
    framecache = []

    f_frames.seek(0, os.SEEK_END)
    txt_sz = hop_sz = f_frames.tell()
    hop_pos = 0
    safe = 0
    sm = 9000  # slingringsmonn
    while hop_sz > sm:
        hop_sz //= 2
        f_frames.seek(hop_pos)
        if hop_pos != 0:
            next(f_frames)

        ofs = int(next(f_frames).split(" ")[1])
        if ofs < lower_byte:
            safe = hop_pos

        if ofs > lower_byte - sm:
            hop_pos = max(0, hop_pos - hop_sz)
        elif ofs < lower_byte - sm * 2:
            hop_pos = min(txt_sz - sm, hop_pos + hop_sz)
        else:
            break

    f_frames.seek(safe)
    if safe != 0:
        next(f_frames)

    sec2 = None
    for ln in f_frames:
        nframe, ofs, sec = ln.split(" ")
        nframe = int(nframe)
        ofs = int(ofs)
        sec = float(sec)
        if ofs < lower_byte:
            continue
        if sec2 is None:
            if ofs >= ofs2:
                sec2 = sec
        else:
            if sec >= sec2 + extra_sec:
                break
        framecache.append([nframe, ofs, sec])

    if not framecache:
        msg = "could not build framecache ofs {}..{}, last seen ofs {}, sec {}"
        raise Exception(msg.format(ofs1, ofs2, ofs, sec))

    return framecache


def find_silence(silents, target_sec, min_lower):
    best_sec_mid = None
    best_sec_end = None
    best_frame = None
    best_dist = None
    best_db = None
    lower_bound = max(min_lower, target_sec - 30)
    upper_bound = max(min_lower, target_sec + 30)
    for db in [60, 50, 40, 30]:
        if db <= 40 and best_dist:
            # skip 40dB if we have anything at all
            break

        if db not in silents:
            continue

        for nframe, t1, t2 in silents[db]:
            if t2 < lower_bound:
                continue

            if t1 > upper_bound:
                break

            t = t1 + (t2 - t1) * 0.8
            dist = abs(target_sec - t)
            if not best_dist or dist < best_dist:
                if not best_db or dist < best_dist / 4:
                    best_dist = dist
                    best_frame = nframe
                    best_sec_mid = t
                    best_sec_end = t2

    if best_sec_mid:
        return [best_sec_mid, best_sec_end, best_frame]
    else:
        return [target_sec, None, None]


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
            fn = "{}/{:03d}. {}.mp3".format(outdir, ntrack, title or "x")
            f = open(fn, "wb")
            break
        except:
            if not title:
                raise
            title = title[:-1]

    f_defrosted.seek(ofs1)
    rem = ofs2 - ofs1
    while rem > 0:
        buf = f_defrosted.read(min(rem, 4096))
        if not buf:
            raise Exception("hit eof in defrosted mp3")

        rem -= len(buf)
        f.write(buf)

    f.close()
    return fn


def tag_mp3(fn, tagtxt, timestr, album):
    artist = None
    ofs = tagtxt.find(" - ")
    if ofs < 0:
        title = tagtxt
    elif ofs == 0:
        title = tagtxt[3:]
    else:
        artist = tagtxt[:ofs]
        title = tagtxt[ofs + 3 :]

    try:
        debug("writing tags")
        comment = "{} UTC, defrost.py".format(timestr)
        timestr = timestr.replace(", ", "T")

        id3 = ID3()
        id3.add(TIT2(encoding=3, text=title))
        if artist:
            id3.add(TPE1(encoding=3, text=artist))

        if album:
            id3.add(TALB(encoding=3, text=album))

        id3.add(TDRC(encoding=3, text=timestr))
        id3.add(COMM(encoding=3, text=comment))
        id3.save(fn)
    except Exception as ex:
        warn("failed to write tags: %r", ex)


def add_sentinel(gen):
    for x in gen:
        yield x

    yield None


def main():
    global YOLO, METAINT

    # fmt: off
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", metavar="FORMAT", default="icy", help="input format; icy=default=icecast, ls=loopstream")
    ap.add_argument("-d", action="store_true", help="enable debug output")
    ap.add_argument("-f", action="store_true", help="overwrite existing split")
    ap.add_argument("-o", metavar="DIR", help="output directory")
    ap.add_argument("-a", metavar="ALBUM", help="album title for id3 tags")
    ap.add_argument("--metaint", type=int, help="icecast metadata interval", default=METAINT)
    ap.add_argument("--no-split", action="store_true", help="do not split the mp3")
    ap.add_argument("--no-id3", action="store_true", help="do not write id3 tags")
    ap.add_argument("--yolo", action="store_true", help="less sanity checks (for buggy files)")
    ap.add_argument("--clean", action="store_true", help="redo most of the cached steps")
    ap.add_argument("ICY_MP3", help="icecast recording with inband metadata")
    ar = ap.parse_args()
    # fmt: on

    if WINDOWS:
        os.system("rem")  # best girl

    YOLO = ar.f
    METAINT = ar.metaint
    configure_logger(ar.d)

    rc = 0
    t0 = time.time()
    fn = ar.ICY_MP3
    sz = os.path.getsize(fn)
    lastmod = os.path.getmtime(fn)
    info("using {} ({:.2f} MiB)".format(fn, sz / 1024.0 / 1024))

    path_hash = hashlib.sha1(fn.encode("utf-8", "ignore")).digest()
    path_hash = base64.urlsafe_b64encode(path_hash)[:8].decode("ascii")

    debug("hashing final 64 MiB")
    hasher = hashlib.sha1()
    with open(fn, "rb", IO_BUFSZ) as f:
        f.seek(max(0, sz - 64 * 1024 * 1024))
        while True:
            buf = f.read(4096)
            if not buf:
                break

            hasher.update(buf)

    file_hash = hasher.digest()
    file_hash = base64.urlsafe_b64encode(file_hash)[:8].decode("ascii")

    base = fn.replace("\\", "/").split("/")[-1]
    localbase = "{}.defrost-{}-{}".format(base, file_hash, sz)
    # localbase += "a"  # cfg: when windows says no

    outdir = ar.o or localbase
    info("using {} for cache".format(localbase))
    info("using {} for outdir".format(outdir))

    fn_mp3 = localbase + ".mp3"
    fn_idx = localbase + ".idx"
    fn_frames = localbase + ".frames"
    fn_silence = localbase + ".silence"
    if ar.clean:
        debug("deleting cache")
        drop = [fn_frames, fn_silence]
        # drop.extend([fn_mp3, fn_idx])
        for f in drop:
            if os.path.exists(f):
                os.unlink(f)

    if ar.f and os.path.exists(outdir):
        debug("deleting old split")
        shutil.rmtree(outdir)

    if not ar.no_split:
        for n in range(3):
            try:
                os.mkdir(outdir)
                break
            except:
                if os.path.isdir(outdir):
                    break

                if n == 2:
                    raise

                time.sleep(0.3)
                pass

    if ar.i == "ls":
        fn_mp3 = fn

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
            if not YOLO:
                time.sleep(2)

        print("\n")
        info("running decode pass")
        run_defrost(fn, fn_mp3, fn_idx, sz)

    if os.path.exists(fn_frames):
        with open(fn_frames, "r", encoding="utf-8") as f:
            ln = next(f)
            fields = ln.split(" ")
            if len(fields) != 3 or fields[0] != "0":
                warn("deleting old %s (format has changed)", fn_frames)
                os.unlink(fn_frames)

    sz = os.path.getsize(fn_mp3)
    if not os.path.exists(fn_frames):
        print("\n")
        info("collecting frames with FFmpeg")
        buf = []
        pos = 0
        nframes = -1
        with open(fn_frames, "w", encoding="utf-8") as f:
            for pos, ts in collect_frames(fn_mp3):
                nframes += 1
                buf.append("%s %s %.3f\n" % (nframes, pos, ts))
                if len(buf) > 1024 * 16:
                    print("\033[A%.2f%% " % (pos * 100.0 / sz,))
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
    with open(fn_frames, "r", IO_BUFSZ, encoding="utf-8") as f:
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
        with open(fn_silence, "w", encoding="utf-8") as f:
            for nframe, db, start, end in r:
                ln = "%s %s %.2f %.2f\n" % (nframe, db, start, end)
                f.write(ln)

    # probably small enough
    print()
    info("reading silent ranges into memory")
    silents = {}  # dB => [nframe, t1, t2]  (prefer timestamps)
    with open(fn_silence, "r", encoding="utf-8") as f:
        for ln in f:
            fields = ln.split(" ")
            nframe, db = [int(x) for x in fields[:2]]
            t1, t2 = [float(x) for x in fields[2:]]
            entry = [nframe, t1, t2]
            try:
                silents[db].append(entry)
            except:
                silents[db] = [entry]

    if ar.i == "ls":
        # {"o": 6592000, "o1": 6608493, "o2": 12816880, "t": "Q2xhcmlTIC0gaXJvbnk=", "z": [16000, 32001, 48002,
        ptn = re.compile(r"^([0-9]{2,8}):([0-9]{2}):([0-9]{2}) (.*)$")
        with open(fn + ".txt", "r", encoding="utf-8", errors="replace") as f_txt, open(fn_frames, "r", encoding="utf-8") as f_frames, open(fn_idx, "w", encoding="utf-8") as f_idx:
            for _ in range(4096):
                ofs = f_txt.tell()
                if f_txt.read(1) in "1234567890":
                    break
            f_txt.seek(ofs)

            for txt_ln in f_txt:
                m = ptn.match(txt_ln)
                if not m:
                    error("bad loopstrema metadata line: " + txt_ln)
                    continue

                g = m.groups()
                s = [int(x) for x in g[:3]]
                s = 60 * (60 * s[0] + s[1]) + s[2]
                t = base64.urlsafe_b64encode(g[3].encode("utf-8")).decode("utf-8")

                # just grab the first frame that fits, assume mp3 not oob
                for fr in f_frames:
                    # 7 4388 0.183
                    _, o, fs = [int(x) for x in fr.split(".")[0].split(" ")]
                    if fs >= s:
                        break

                f_idx.write(json.dumps({"o": o, "t": t}) + "\n")

    ntrack = 0
    framecache = []  # nframe, ofs, sec
    with open(fn_idx, "r", encoding="utf-8") as f_idx, open(fn_frames, "r", encoding="utf-8") as f_frames, open(
        fn_mp3, "rb"
    ) as f_defrosted:
        tag = None
        next_sec = None  # position (in absolute seconds) that was last returned
        for idx_ln in add_sentinel(f_idx):
            if idx_ln:
                tag2 = json.loads(idx_ln)
                tag2 = {
                    "o": tag2["o"],
                    "t": base64.urlsafe_b64decode(tag2["t"].encode("ascii")),
                }
                if not tag:
                    tag = tag2
                    next_sec = 0
                    continue
            else:
                tag2 = {"o": sz}  # read until eof

            ntrack += 1
            ofs1 = tag["o"]
            ofs2 = tag2["o"]
            perc = ofs1 * 100.0 / sz
            debug("%.0f%% track #%s, ofs %s..%s", perc, ntrack, ofs1, ofs2)

            # 38 frames in 1 sec (1152 samples per frame),
            # 192k=627b, 256k=836bm 320k=1045b
            lower = max(ofs1 - 320 * 128 * 60, 0)
            upper = min(ofs2 + 320 * 128 * 60, sz)
            if framecache and framecache[0][0] == 0:
                lower = framecache[0][1]  # anchored to start of file already

            if not framecache or framecache[0][1] > lower or framecache[-1][1] < upper:
                if framecache:
                    msg = "framecache insufficient, {:,} > {:,} or {:,} < {:,}"
                    debug(msg.format(framecache[0][1], lower, framecache[-1][1], upper))
                framecache = build_framecache(f_frames, ofs1, ofs2)
                msg = "framecache covers frame {}..{}, ofs {:,}..{:,}, ts {}..{}"
                fc0 = framecache[0]
                fc1 = framecache[-1]
                debug(msg.format(fc0[0], fc1[0], fc0[1], fc1[1], fc0[2], fc1[2]))
                if fc0[1] > ofs1 + 4096 or fc1[1] < ofs2 - 4096:
                    msg = "framecache still insufficient, {:,} > {:,} or {:,} < {:,}"
                    raise Exception(msg.format(fc0[1], ofs1, fc1[1], ofs2))

            tag_ts1 = None
            tag_ts2 = None
            frame1 = None
            frame2 = None
            for nframe, ofs, ts in framecache:
                # print("cmp {} {}".format(ofs, ofs2))
                if tag_ts1 is None:
                    if ts >= next_sec:
                        frame1 = nframe
                        tag_ts1 = ts
                        ofs1 = ofs
                elif tag_ts2 is None or ofs <= ofs2:
                    frame2 = nframe
                    tag_ts2 = ts
                else:
                    break

            if tag_ts1 is None:
                a, b, c = framecache[0]
                msg = "could not find next_sec {} in framecache, min is {} (#{}, {}b), gave up at {} (#{}, {}b)"
                raise Exception(msg.format(next_sec, c, a, b, ts, nframe, ofs))

            msg = "%.0f%% track #%s, ofs %s..%s, sec %s..%s, frame %s..%s"
            debug(msg, perc, ntrack, ofs1, ofs2, tag_ts1, tag_ts2, frame1, frame2)

            ts1 = tag_ts1
            ts2 = tag_ts2
            if idx_ln:
                quiet_ts2, quiet_ts_ref, quiet_frame2 = find_silence(
                    silents, tag_ts2, next_sec
                )
            else:
                quiet_ts2 = tag_ts2
                quiet_ts_ref = tag_ts2
                quiet_frame2 = frame2

            for nframe, ofs, ts in framecache:
                if ts > quiet_ts2:
                    break

                frame2 = nframe
                ofs2 = ofs
                ts2 = ts

            # ffmpeg does some sick drifts occasionally
            # so lets be extremely lenient with ffmpeg/ffprobe differences
            checks = [
                ["quiet2", quiet_frame2, quiet_ts_ref],
                ["fc1", frame1, ts1],
                ["fc2", frame2, ts2],
            ]
            results = []
            for _, frame, ts in checks:
                if not frame or frame < 10 or ts < 1:
                    continue
                results.append(frame / ts)

            results.sort()
            # bump threshold below if necessary, just guessing
            if results and results[-1] / results[0] >= 1.05:
                msg = "something desynced:\n{}".format(
                    pprint.pformat([checks, results])
                )
                if YOLO:
                    error(msg)
                else:
                    raise Exception(msg)

            msg = "{:.0f}% track #{}, ofs {}..{}, sec {}..{}, cut {}..{}, delta {:.2f}..{:.2f}"
            msg = msg.format(
                perc,
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
            debug("%.0f%% tag: [%s] [%s]", perc, enc, tagtxt)

            if not ar.no_split:
                unix = int(lastmod - (end_ts - ts1))
                if ar.i == "ls":
                    ptn = r"([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}\.[0-9]{2}\.[0-9]{2})"
                    m = re.search(ptn, os.path.basename(ar.ICY_MP3))
                    if m:
                        dt = time.strptime(m.group(1), "%Y-%m-%d_%H.%M.%S")
                        unix = int(calendar.timegm(dt)) + ts1
                fmt = "%Y-%m-%d, %H:%M:%S"
                timestr = datetime.fromtimestamp(unix, UTC).strftime(fmt)
                fn = split_mp3(f_defrosted, ntrack, ofs1, ofs2, outdir, tagtxt)
                if not ar.no_id3:
                    tag_mp3(fn, tagtxt, timestr, ar.a)

                os.utime(fn, (int(time.time()), unix))
                info("{:.0f}% wrote [{}] [{}]".format(perc, timestr, fn))

            next_sec = ts2
            tag = tag2

    print()
    tmp_files = [fn_mp3, fn_idx, fn_frames, fn_silence]
    msg = "finished in {:.2f} sec with {} errors\nyou can delete these now:\n  {}"
    msg = msg.format(time.time() - t0, rc or "no", "\n  ".join(tmp_files))
    fun = error if rc else info
    fun(msg)
    sys.exit(rc)


if __name__ == "__main__":
    main()


# c:\Users\ed\bin\pypy3\pypy3.exe defrost.py 5g-ok.mp3
# c:\Users\ed\bin\pypy2\pypy.exe defrost.py 5g-ok.mp3
# c:\Python27\python.exe defrost.py 5g-ok.mp3
#
#        pypy2  pypy3   cpy2   cpy3
# win32  25.80  14.57
# win64                20.57  20.18
# linux  14.78  65.55  28.87   8.35


# printf '\n\n'; for n in {1..1024}; do printf '\033[2A%d\n' $n; cmp <(tail -c +$((909062+n)) <tailed.mp3.defrost-2p0U8fyy-10485760.mp3) tailed.mp3.defrost-2p0U8fyy-10485760/2.\ FLOW\ -\ COLORS.mp3 2>&1 | tee /dev/stderr | grep EOF && break; done
#
# cmp <(tail -c +$((145+909062+5271301+2056359+126015)) <tailed.mp3.defrost-2p0U8fyy-10485760.mp3) tailed.mp3.defrost-2p0U8fyy-10485760/5.\ s3rl\ -\ MTC.mp3
# cmp: EOF on tailed.mp3.defrost-2p0U8fyy-10485760/5. s3rl - MTC.mp3 after byte 2121561, in line 9387


# cat -n 5g-ok.mp3.defrost-z66AMwbk-5370838524.idx | sed -r 's/", "z": .*//; s/(.*), "t": "(.*)/\2 \1/' | while read b64 v; do printf '%s %s\n' "$v" "$(printf '%s\n' "$b64" | base64 -d 2>/dev/null)"; done


########################################################################
## end of debug notes,
## start of usage notes


##
## convert a folder of loopstream recordings starting at file "Loopstream-2021-10-31_21.57.26.mp3"

# en=; for f in *.mp3; do [ "$f" = "Loopstream-2021-10-31_21.57.26.mp3" ] && en=1; [ $en ] || continue; t=${f:11:4}-${f:16:2}${f:19:2}-${f:22:2}${f:25:2}${f:28:2}; echo $t; PYTHONPATH=/c/Users/ed/dev/defrost/ python -m defrost -i ls -o $t "$f"; done; rm -- *.mp3.defrost-*


##
## get all dj changes and songs played from znc irc logs

# grep -iE '^[^> ]+ [^> ]+> \.dj |^[^> ]+ <Hanyuu-sama> Now Starting:.04 ' 2021-05-0* | sed -r 's/, .03LP:. [0-9].*//; s/> Now starting:.04 (.*)(\[[0-9]{2}:[0-9]{2}\])/'$'> \033[33m\\1\033[36m\\2/; s/$/\033[0m/'

##
## listing all songs and their offsets in a recording

# alternative 1, 45 MiB/s:
#   strings -td -eS -n 13 main.mp3 | grep StreamTitle=
#
# alternative 2, 200 MiB/s:
#   bgrep -A300 $(echo -n StreamTitle= | xxd -p) ../main.mp3.6 | awk "/^StreamTitle=/{sub(/';\\\\x.*/,"'"");print "0x" o,$0} {o=$2}' | while read o t; do printf '%d %s\n' $o "$t"; done
#
# 16001 StreamTitle='Rita - Alicemagic (Tsukasa Remix)';
# 7216515 StreamTitle='Mizuki Nana - METANOIA';

##
## discarding everything before a given song

# function slice() { tail -c+$(($3-16000)) <$1 >$2; touch -r $1 $2; }
# slice main.mp3 slice.mp3 7216515

##
## taking the ~30 last hours of a file

# tail -c $((1024*1024*85*30)) main.mp3.7 >tmp.mp3; touch -r main.mp3.7 /dev/shm/tmp.mp3
# python3 defrost.py tmp.mp3
# "Exception: initial tag at 48ba29, expected at 3e80"
# d=$(printf %d 0x48ba29); tail -c +$((d-15999)) <tmp.mp3 >tmp2.mp3; touch -r main.mp3.7 tmp2.mp3
