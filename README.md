# defrost
* split broken icecast recordings into separate mp3s
* MIT-Licensed, 2019-06-01, ed @ irc.rizon.net, [github](https://github.com/9001/defrost), [pypi](https://pypi.org/project/defrostir/)

# features
* **silence detection** -- finds the best part to split at
* **mp3 parser** -- clean splits at frame boundaries
* **icy-meta parser** -- extract artist/title from inband metadata
  * dynamic recalibration to recover from server glitches
* **charset detection** -- UTF8, latin-1, shift-jis and friends, all good
* **ID3 tags** -- uses mutagen to write aritst/title into the files
* **timestamping** -- the date/time that the song was played live
* support for other input formats:
  * loopstream recording

# installation
* get the latest release from pypi: `python3 -m pip install --user -U defrostir`
* or the latest commit from github: `python3 -m pip install --user -U https://github.com/9001/defrost/tarball/hovudstraum`


# usage

from an internet radio station:
```
wget -U MPlayer --header "Icy-MetaData: 1" -S "https://stream.r-a-d.io/main.mp3"
# wait until the show is over, then abort the download
python3 -m defrostir main.mp3
```

from a loopstream recording:
```
python3 -m defrostir -i ls Loopstream-2021-10-02_22.53.44.mp3
```

# notes
versions `x.y.1` have [chardet](https://pypi.org/project/chardet/) as a dependency, while `x.y.2` have [charset-normalizer](https://pypi.org/project/charset-normalizer/) (preferred) and are otherwise identical; `setup.py rls` will produce both to avoid a pypi package selection bug
