#!/usr/bin/env python2
from __future__ import print_function

import os
import sys
import codecs
import struct
import unicodedata

# mkdir tags; for f in main.mp3*; do python3 -u ~/dev/icychk.py "$f" 2>&1 | tee "tags/$f.tags"; echo "${PIPESTATUS[*]}" >> "tags/$f.tags"; done
# for x in tags/*; do printf '\n%s\n' "$x"; tail -n 2 "$x"; done
# for x in *; do printf '\n\n%s' "$x"; awk '{print $1}' "tags/$x.tags" | while IFS= read -r ofs; do printf '\n%s ' $ofs; dd if="$x" bs=1 count=13 skip=$ofs 2>/dev/null; done; done | tee /dev/shm/log

METAINT = 16000
IO_BUFSZ = 512 * 1024


if sys.version_info[0] > 2:
    uprint = print
else:
    UPRINT = codecs.getwriter("utf-8")

    def uprint(*args, **kwargs):
        kwargs["file"] = UPRINT(sys.stdout)
        print(*list(args), **kwargs)


class Verifier(object):
    """
    strict parser, assumes correct metaint, raises on corruption
    (see Metascanner for the lenient version of this)
    """

    def __init__(self, f):
        self.f = f
        self.find = b"StreamTitle='"
        self.icystep = METAINT + 1
        self.run()

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
                raise Exception(u"bad magic @ {} / {:x}".format(ofs, ofs))

            try:
                meta = buf.decode("utf-8")
            except:
                meta = buf.decode("utf-8", "ignore") + u" INVAL"

            meta2 = u"".join(c for c in meta if unicodedata.category(c)[0] != "C")

            if meta != meta2:
                meta = meta2 + u" ESC"
                # print(u'[{}]\n[{}]'.format(meta,meta2))

            uprint(u"{} {}".format(ofs - 1, meta))


class Metascanner(object):
    """
    find metadata without relying on metaint
    (use this instead of Verifier to determine metadata location)
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
    err = u""
    try:
        meta = buf.decode("utf-8")
    except:
        meta = buf.decode("utf-8", "ignore")
        err += u" MOJIBAKE"

    safe_meta = u"".join(c for c in meta if unicodedata.category(c)[0] != "C")

    if meta != safe_meta:
        meta = safe_meta
        err += u" CONTROL"
        # print(u'[{}]\n[{}]'.format(meta,meta2))

    return [meta, err]


def defrost(fn, msc, zcf):
    """
    yields [lower, upper, zeros, tags]
    """
    yield_ofs = 0
    zeros = []
    tags = []
    ofs1 = 0
    bm1 = None
    # iterate over all detected tags,
    # consider ranges between previous and current tag
    for ofs2, bm2 in msc.run():
        tags.append(bm2)
        uprint(u"{0} {0:x} {1}{2}".format(ofs2, *tag2text(bm2)))

        if not bm1:
            if ofs2 != METAINT:
                raise Exception(
                    "initial tag at {:x}, expected at {:x}".format(ofs2, METAINT)
                )
            zeros_up = []
            zeros_dn = zerochk(zcf, 0, ofs2, False)
        else:
            after_tag = ofs1 + len(bm1) + 1
            zeros_up = zerochk(zcf, after_tag, ofs2, True)
            zeros_dn = zerochk(zcf, after_tag, ofs2, False)

            # add the tag itself as a range
            zeros.extend([ofs1, after_tag])

        if bm1 and ofs2 - ofs1 == METAINT + len(bm1) + 1:
            # immediate tag change with correct metaint spacing
            pass
        elif zeros_up and zeros_up == zeros_dn:
            # another best case scenario; no corruption
            zeros.extend(zeros_up)
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
                    "\033[35mUP {}".format(" ".join("{:x}".format(x) for x in zeros_up))
                )

            gap_upper = ofs2
            if zeros_dn:
                gap_upper = zeros_dn[0]
                print(
                    "\033[33mDN {}".format(" ".join("{:x}".format(x) for x in zeros_dn))
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

            yield [yield_ofs, gap_lower + METAINT, zeros, tags]
            yield_ofs = gap_upper
            tags = tags[-1:]
            zeros = []

            if zeros_dn:
                zeros.extend(zeros_dn)

        ofs1 = ofs2
        bm1 = bm2

    # final readout
    fsz = os.path.getsize(fn)
    after_tag = ofs2 + len(bm2) + 1
    zeros.extend([ofs2, after_tag])
    zeros.extend(zerochk(zcf, after_tag, fsz, True))
    yield [yield_ofs, fsz, zeros, tags]


def main():
    if len(sys.argv) <= 1:
        print("usage: {} some.mp3".format(sys.argv[0]))
        sys.exit(1)

    fn = sys.argv[1]
    print("using {0}".format(fn))
    with Metascanner(fn) as metascanner:
        with open(fn, "rb", IO_BUFSZ) as zerochk_fd:
            for lower, upper, zeros, tags in defrost(fn, metascanner, zerochk_fd):
                uprint(
                    u"l {:x}   u {:x}   z {}   t {}".format(
                        lower, upper, len(zeros), len(tags)
                    )
                )

    # print(u"{} {}{}".format(ofs2, meta, err))


if __name__ == "__main__":
    main()
